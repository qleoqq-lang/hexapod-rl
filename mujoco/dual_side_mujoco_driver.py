"""External MuJoCo replacement for dual_side_control's CAN/MIT boundary."""

from __future__ import annotations

import math
import json
import shutil
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mujoco
import mujoco.viewer
import numpy as np


LEG_ORDER = ["FL", "ML", "RL", "FR", "MR", "RR"]
JOINT_NAMES = ["Coxa", "Femur", "Tibia"]
JOINT_SUFFIXES = ["coxa", "femur", "tibia"]
MJCF_PREFIX = {"FL": "lf", "ML": "lm", "RL": "lr", "FR": "rf", "MR": "rm", "RR": "rr"}
COMMAND_TO_MJCF_SIGNS = {
    "FL": (-1.0, 1.0, -1.0),
    "ML": (-1.0, 1.0, -1.0),
    "RL": (-1.0, 1.0, -1.0),
    "FR": (-1.0, -1.0, -1.0),
    "MR": (-1.0, 1.0, -1.0),
    "RR": (-1.0, 1.0, -1.0),
}


@dataclass
class MotorConfig:
    name: str
    can_id: int
    mst_id: int
    init_kp: float
    init_kd: float
    kp_min: float
    kp_max: float
    kd_min: float
    kd_max: float
    pos_min: float
    pos_max: float


@dataclass
class MotorState:
    motor_id: int
    position: float
    velocity: float
    torque: float
    enabled: bool


@dataclass
class JointDirectionMap:
    legs: Dict[str, List[int]]
    confirmed: bool = False
    updated_at: str = ""
    source: str = "default"

    def ensure_complete(self) -> None:
        for leg in LEG_ORDER:
            values = self.legs.get(leg, [1, 1, 1])
            if len(values) != 3:
                values = [1, 1, 1]
            self.legs[leg] = [1 if value >= 0 else -1 for value in values]


class DirectionMapStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> JointDirectionMap:
        if not self.path.exists():
            return JointDirectionMap({leg: [1, 1, 1] for leg in LEG_ORDER})
        with self.path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        result = JointDirectionMap(
            legs={leg: list(values) for leg, values in payload.get("legs", {}).items()},
            confirmed=bool(payload.get("confirmed", False)),
            updated_at=str(payload.get("updated_at", "")),
            source=str(payload.get("source", "default")),
        )
        result.ensure_complete()
        return result

    def save(self, direction_map) -> None:
        direction_map.ensure_complete()
        payload = {
            "confirmed": bool(direction_map.confirmed),
            "updated_at": str(direction_map.updated_at or time.strftime("%Y-%m-%d %H:%M:%S")),
            "source": str(direction_map.source),
            "legs": direction_map.legs,
        }
        with self.path.open("w", encoding="utf-8") as destination:
            json.dump(payload, destination, ensure_ascii=False, indent=2)


def transform_joint_angles(leg_name: str, values: Sequence[float],
                           direction_map: JointDirectionMap) -> List[float]:
    signs = direction_map.legs.get(leg_name, [1, 1, 1])
    return [float(value) * signs[index] for index, value in enumerate(values)]


@dataclass
class MITTarget:
    kp: float = 0.0
    kd: float = 0.0
    position: float = 0.0
    velocity: float = 0.0
    feedforward_torque: float = 0.0
    enabled: bool = False


def load_side_config(path: str) -> tuple[List[MotorConfig], Dict[str, List[float]]]:
    root = ET.parse(path).getroot()

    def number(parent: ET.Element, tag: str, default: float) -> float:
        text = parent.findtext(tag)
        return float(text) if text else default

    motors = []
    for index, element in enumerate(root.findall("motor")):
        motors.append(MotorConfig(
            name=JOINT_NAMES[index % 3],
            can_id=int(element.findtext("can_id", "0"), 16),
            mst_id=int(element.findtext("mst_id", "0"), 16),
            init_kp=number(element, "init_kp", 20.0),
            init_kd=number(element, "init_kd", 0.8),
            kp_min=number(element, "kp_min", 1.0),
            kp_max=number(element, "kp_max", 20.0),
            kd_min=number(element, "kd_min", 0.0),
            kd_max=number(element, "kd_max", 2.0),
            pos_min=number(element, "joint_min", -math.pi),
            pos_max=number(element, "joint_max", math.pi),
        ))
    presets = {}
    presets_root = root.find("presets")
    if presets_root is not None:
        for preset in presets_root:
            if preset.text:
                values = [float(value.strip()) for value in preset.text.split(",")]
                if len(values) == 3:
                    presets[preset.tag] = values
    return motors, presets


def load_model(path: str) -> mujoco.MjModel:
    source = Path(path).resolve()
    try:
        return mujoco.MjModel.from_xml_path(str(source))
    except ValueError as direct_error:
        meshes = source.parent.parent / "meshes"
        if not meshes.is_dir():
            raise direct_error
        with tempfile.TemporaryDirectory(prefix="hexapod_control_") as temporary:
            staging = Path(temporary)
            staged_mujoco = staging / "mujoco"
            staged_mujoco.mkdir()
            staged_model = staged_mujoco / source.name
            shutil.copy2(source, staged_model)
            shutil.copytree(meshes, staging / "meshes")
            return mujoco.MjModel.from_xml_path(str(staged_model))


class MuJoCoDualSideDriver:
    """Public API compatible with the original ``DualSideMITDriver``."""

    model_path: Optional[str] = None
    control_root: Optional[str] = None
    show_viewer = True
    auto_enable_on_start = False
    peak_torque_nm = 7.0
    control_period_s = 0.002  # com_layer.cpp: ControlLoop / InitializeRuckig
    viewer_refresh_period_s = 1.0 / 60.0
    gain_limit_policy = "motor"
    motor_kp_limits = (0.0, 500.0)
    motor_kd_limits = (0.0, 5.0)

    def __init__(self, left_config: str, right_config: str):
        if not self.model_path:
            raise RuntimeError("MuJoCo model path was not configured by the external launcher")
        if not self.control_root:
            raise RuntimeError("dual_side_control path was not configured by the external launcher")
        try:
            import _original_ruckig
        except ImportError as error:
            raise RuntimeError(
                "Original robot Ruckig module is missing or incompatible. "
                "Build it with mujoco/native/build.ps1 using this Python environment. "
                "Direct-target fallback is deliberately disabled."
            ) from error
        self._ruckig_module = _original_ruckig
        left_motors, left_presets = load_side_config(left_config)
        right_motors, right_presets = load_side_config(right_config)
        if len(left_motors) != 9 or len(right_motors) != 9:
            raise RuntimeError("Expected 9 motors in each original XML config")

        self.motors = left_motors + right_motors
        self.side_presets = (left_presets, right_presets)
        self.num_legs = 6
        self.num_motors = 18
        self.presets = sorted(set(left_presets) & set(right_presets))
        self.direction_store = DirectionMapStore(
            Path(self.control_root) / "config" / "joint_direction_map.json"
        )
        self.direction_map = self.direction_store.load()

        self.model = load_model(self.model_path)
        if not math.isclose(self.model.opt.timestep, self.control_period_s, abs_tol=1e-12):
            raise ValueError("Original control loop requires an MJCF timestep of 0.002 s")
        self.data = mujoco.MjData(self.model)
        if self.model.nu != 18:
            raise RuntimeError(f"MJCF must have 18 actuators; got {self.model.nu}")
        stand_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        if stand_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, stand_key)
            mujoco.mj_forward(self.model, self.data)

        names = [f"{MJCF_PREFIX[leg]}_{joint}_joint" for leg in LEG_ORDER for joint in JOINT_SUFFIXES]
        self.joint_ids = []
        self.actuator_ids = []
        self.command_signs = np.array([
            sign for leg in LEG_ORDER for sign in COMMAND_TO_MJCF_SIGNS[leg]
        ])
        for name in names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
            if joint_id < 0 or actuator_id < 0:
                raise RuntimeError(f"Missing joint/actuator mapping: {name}")
            self.joint_ids.append(joint_id)
            self.actuator_ids.append(actuator_id)

        self.lock = threading.RLock()
        self.commands = [MITTarget() for _ in range(18)]
        self.applied_targets = np.zeros(18)
        self.applied_velocities = np.zeros(18)
        self._pending_ruckig = [None] * 6
        self._active_ruckig = [None] * 6
        self.zero_offsets = np.zeros(18)
        self.running = True
        self.stop_event = threading.Event()
        self.simulation_error: Optional[BaseException] = None
        self.viewer = None
        self.viewer_owner_thread_id: Optional[int] = None
        self._viewer_last_wall_time: Optional[float] = None
        self._viewer_step_accumulator_s = 0.0
        self.thread: Optional[threading.Thread] = None
        for index, joint_id in enumerate(self.joint_ids):
            q_address = int(self.model.jnt_qposadr[joint_id])
            initial_position = (
                float(self.data.qpos[q_address]) * self.command_signs[index]
            )
            self.applied_targets[index] = initial_position
            if self.auto_enable_on_start:
                self._write(index, initial_position, enabled=True)
        if self.show_viewer:
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("MuJoCo Viewer must be created on the main thread")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer_owner_thread_id = threading.get_ident()
            self._viewer_last_wall_time = time.perf_counter()
        else:
            self.thread = threading.Thread(target=self._run_simulation, daemon=True)
            self.thread.start()

    def _step_simulation_locked(self) -> None:
        # Same ordering as com_layer.cpp: new request -> Ruckig update -> MIT.
        smooth_states = self._update_ruckig_locked()
        for index, command in enumerate(self.commands):
            joint_id = self.joint_ids[index]
            actuator_id = self.actuator_ids[index]
            q_address = int(self.model.jnt_qposadr[joint_id])
            v_address = int(self.model.jnt_dofadr[joint_id])
            if not command.enabled:
                self.data.ctrl[actuator_id] = 0.0
                self.applied_targets[index] = (
                    self.data.qpos[q_address] - self.zero_offsets[index]
                ) * self.command_signs[index]
                self.applied_velocities[index] = 0.0
                continue
            smooth = smooth_states[index // 3]
            # Active Ruckig has priority over ordinary MIT streaming, just as
            # in the robot. The walking executor uses the ordinary MIT path.
            self.applied_targets[index] = (
                command.position if smooth is None else smooth[0][index % 3]
            )
            self.applied_velocities[index] = (
                command.velocity if smooth is None else smooth[1][index % 3]
            )
            physical_target = (
                self.applied_targets[index] * self.command_signs[index]
                + self.zero_offsets[index]
            )
            torque = command.kp * (physical_target - self.data.qpos[q_address])
            torque += command.kd * (
                self.applied_velocities[index] * self.command_signs[index]
                - self.data.qvel[v_address]
            )
            torque += command.feedforward_torque * self.command_signs[index]
            self.data.ctrl[actuator_id] = np.clip(
                torque, -self.peak_torque_nm, self.peak_torque_nm
            )
        mujoco.mj_step(self.model, self.data)

    def _update_ruckig_locked(self):
        """Sample the unchanged robot C++ wrapper once per 2 ms control tick."""
        outputs = [None] * 6
        for leg_index in range(6):
            target = self._pending_ruckig[leg_index]
            if target is not None:
                self._pending_ruckig[leg_index] = None
                indices = self._indices(leg_index)
                current = self._ruckig_module.JointState()
                current.position = [
                    (float(self.data.qpos[self.model.jnt_qposadr[self.joint_ids[i]]])
                     - self.zero_offsets[i]) * self.command_signs[i]
                    for i in indices
                ]
                current.velocity = [
                    float(self.data.qvel[self.model.jnt_dofadr[self.joint_ids[i]]])
                    * self.command_signs[i] for i in indices
                ]
                current.acceleration = [0.0] * 3  # Same measured-state input as C++.
                trajectory = self._ruckig_module.RuckigTrajectory(self.control_period_s)
                if not trajectory.plan(current, target):
                    # Fail visibly; never silently replace a failed plan by a jump.
                    raise RuntimeError(f"Original Ruckig planning failed for {LEG_ORDER[leg_index]}")
                self._active_ruckig[leg_index] = trajectory
            trajectory = self._active_ruckig[leg_index]
            if trajectory is not None:
                has_more, positions, velocities, _ = trajectory.next_state()
                if has_more:
                    outputs[leg_index] = (positions, velocities)
                else:
                    # C++ clears 'active' before sending this tick's final MIT command.
                    self._active_ruckig[leg_index] = None
        return outputs

    def _run_simulation(self) -> None:
        try:
            while self.running:
                started = time.perf_counter()
                with self.lock:
                    self._step_simulation_locked()
                remaining = self.model.opt.timestep - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except BaseException as error:
            self.simulation_error = error
            self.request_stop()

    def _check_viewer_thread(self) -> None:
        if self.viewer_owner_thread_id != threading.get_ident():
            raise RuntimeError("MuJoCo Viewer API must run on its creating thread")

    def step_viewer(self) -> bool:
        """Advance real-time physics and synchronize one rendered frame."""
        if self.viewer is None:
            raise RuntimeError("MuJoCo Viewer is not active")
        self._check_viewer_thread()
        if not self.running or not self.viewer.is_running():
            return False
        started = time.perf_counter()
        if self._viewer_last_wall_time is None:
            self._viewer_last_wall_time = started
        elapsed = max(0.0, started - self._viewer_last_wall_time)
        self._viewer_last_wall_time = started
        self._viewer_step_accumulator_s += elapsed
        physics_steps = int(
            self._viewer_step_accumulator_s / self.model.opt.timestep
        )
        self._viewer_step_accumulator_s -= (
            physics_steps * self.model.opt.timestep
        )
        try:
            with self.lock:
                # Rendering is normally much slower than the 2 ms robot control
                # period. Advance all physics ticks that elapsed since the last
                # rendered frame so viewer and headless modes share real-time
                # control semantics.
                for _ in range(physics_steps):
                    self._step_simulation_locked()
                self.viewer.sync()
        except BaseException as error:
            self.simulation_error = error
            self.request_stop()
            raise
        remaining = self.viewer_refresh_period_s - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        return True

    def request_stop(self) -> None:
        self.running = False
        self.stop_event.set()

    def wait(self, timeout: float) -> bool:
        """Wait interruptibly; return False when shutdown was requested."""
        self._check()
        stopped = self.stop_event.wait(timeout)
        self._check()
        return not stopped

    def _check(self) -> None:
        if self.simulation_error:
            raise RuntimeError("MuJoCo backend stopped unexpectedly") from self.simulation_error

    @staticmethod
    def _indices(leg_index: int) -> range:
        if not 0 <= leg_index < 6:
            raise ValueError(f"Invalid leg index: {leg_index}")
        return range(leg_index * 3, leg_index * 3 + 3)

    def leg_name(self, leg_index: int) -> str:
        return LEG_ORDER[leg_index]

    def refresh_direction_map(self) -> JointDirectionMap:
        self.direction_map = self.direction_store.load()
        return self.direction_map

    def set_direction_map(self, direction_map: JointDirectionMap) -> None:
        direction_map.ensure_complete()
        self.direction_map = direction_map
        self.direction_store.save(direction_map)

    def direction_confirmed(self) -> bool:
        return self.direction_map.confirmed

    def transform_positions(self, leg_index: int, values: Sequence[float]) -> List[float]:
        return transform_joint_angles(self.leg_name(leg_index), values, self.direction_map)

    transform_values = transform_positions

    def get_leg_motor_configs(self, leg_index: int) -> List[MotorConfig]:
        return [self.motors[index] for index in self._indices(leg_index)]

    def get_leg_presets(self, leg_index: int) -> Dict[str, List[float]]:
        return dict(self.side_presets[0 if leg_index < 3 else 1])

    def get_preset_pose_map(self, preset_name: str) -> Dict[int, List[float]]:
        return {index: list(self.get_leg_presets(index)[preset_name]) for index in range(6)}

    def _write(self, index: int, position: float, velocity: float = 0.0,
               kp: Optional[float] = None, kd: Optional[float] = None,
               torque: float = 0.0, enabled: bool = True) -> None:
        config = self.motors[index]
        if self.gain_limit_policy == "motor":
            kp_min, kp_max = self.motor_kp_limits
            kd_min, kd_max = self.motor_kd_limits
        elif self.gain_limit_policy == "config":
            kp_min, kp_max = config.kp_min, config.kp_max
            kd_min, kd_max = config.kd_min, config.kd_max
        else:
            raise ValueError(
                f"Unknown gain limit policy: {self.gain_limit_policy!r}"
            )
        self.commands[index] = MITTarget(
            kp=float(np.clip(config.init_kp if kp is None else kp, kp_min, kp_max)),
            kd=float(np.clip(config.init_kd if kd is None else kd, kd_min, kd_max)),
            # The original MIT driver writes the requested position, without an
            # XML joint-limit clamp or a second trajectory/slew filter here.
            position=float(position),
            velocity=float(velocity),
            feedforward_torque=float(np.clip(torque, -self.peak_torque_nm, self.peak_torque_nm)),
            enabled=enabled,
        )
        if index % 3 == 0:
            self._pending_ruckig[index // 3] = None

    @staticmethod
    def _gains(value, defaults: Sequence[float]) -> List[float]:
        if value is None:
            return list(defaults)
        if isinstance(value, (float, int)):
            return [float(value)] * 3
        if len(value) != 3:
            raise ValueError("Gain must contain three values")
        return [float(item) for item in value]

    def read_states(self) -> List[MotorState]:
        self._check()
        states = []
        with self.lock:
            for index, config in enumerate(self.motors):
                joint_id = self.joint_ids[index]
                actuator_id = self.actuator_ids[index]
                q_address = int(self.model.jnt_qposadr[joint_id])
                v_address = int(self.model.jnt_dofadr[joint_id])
                states.append(MotorState(
                    motor_id=config.can_id,
                    position=(
                        float(self.data.qpos[q_address]) - self.zero_offsets[index]
                    ) * self.command_signs[index],
                    velocity=float(self.data.qvel[v_address]) * self.command_signs[index],
                    torque=float(self.data.actuator_force[actuator_id]) * self.command_signs[index],
                    enabled=self.commands[index].enabled,
                ))
        return states

    def get_leg_state_degrees(self, leg_index: int) -> List[float]:
        states = self.read_states()
        return [math.degrees(states[index].position) for index in self._indices(leg_index)]

    def enable_all(self) -> None:
        self.enable_all_sequential(0.0)

    def enable_all_sequential(self, delay_s: float = 0.02) -> None:
        states = self.read_states()
        for index, state in enumerate(states):
            with self.lock:
                self._write(index, state.position)
                self.applied_targets[index] = state.position
            time.sleep(delay_s)

    def disable_all(self) -> None:
        with self.lock:
            self._pending_ruckig = [None] * 6
            for command in self.commands:
                command.enabled = False

    def enable_leg(self, leg_index: int) -> None:
        states = self.read_states()
        with self.lock:
            for index in self._indices(leg_index):
                self._write(index, states[index].position)
                self.applied_targets[index] = states[index].position

    def disable_leg(self, leg_index: int) -> None:
        with self.lock:
            self._pending_ruckig[leg_index] = None
            for index in self._indices(leg_index):
                self.commands[index].enabled = False

    def set_leg_zero(self, leg_index: int) -> None:
        with self.lock:
            self._pending_ruckig[leg_index] = None
            for index in self._indices(leg_index):
                q_address = int(self.model.jnt_qposadr[self.joint_ids[index]])
                self.zero_offsets[index] = self.data.qpos[q_address]
                self.commands[index].position = 0.0
                self.applied_targets[index] = 0.0

    def set_zero_all(self) -> None:
        for leg_index in range(6):
            self.set_leg_zero(leg_index)

    def send_leg_command(self, leg_index: int, positions_deg: Sequence[float], kp=None, kd=None) -> None:
        if len(positions_deg) != 3:
            raise ValueError("Each leg requires three target angles")
        positions = self.transform_positions(leg_index, positions_deg)
        configs = self.get_leg_motor_configs(leg_index)
        kps = self._gains(kp, [config.init_kp for config in configs])
        kds = self._gains(kd, [config.init_kd for config in configs])
        with self.lock:
            for local, index in enumerate(self._indices(leg_index)):
                self._write(index, math.radians(positions[local]), kp=kps[local], kd=kds[local])

    def move_leg_smooth(self, leg_index: int, positions_deg: Sequence[float], kp=None, kd=None) -> None:
        with self.lock:
            self.send_leg_command(leg_index, positions_deg, kp, kd)
            # Like use_ruckig + ruckig_target_pos in the original shared-memory
            # command: capture measured state on the next control tick, not here.
            self._pending_ruckig[leg_index] = [
                self.commands[index].position for index in self._indices(leg_index)
            ]

    def apply_pose_all(self, poses: Dict[int, Sequence[float]]) -> None:
        for leg_index in range(6):
            self.move_leg_smooth(leg_index, poses[leg_index])

    def send_pose_all_streaming(self, poses: Dict[int, Sequence[float]],
                                velocity_by_leg: Optional[Dict[int, Sequence[float]]] = None,
                                apply_direction_map: bool = True) -> None:
        with self.lock:
            for leg_index in range(6):
                positions = list(poses[leg_index])
                velocities = ([0.0] * 3 if velocity_by_leg is None
                              else list(velocity_by_leg.get(leg_index, [0.0] * 3)))
                if apply_direction_map:
                    positions = self.transform_positions(leg_index, positions)
                    velocities = self.transform_values(leg_index, velocities)
                for local, index in enumerate(self._indices(leg_index)):
                    self._write(index, math.radians(positions[local]), math.radians(velocities[local]))

    def close(self) -> None:
        self.disable_all()
        self.request_stop()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if self.viewer is not None:
            self._check_viewer_thread()
            self.viewer.close()
            self.viewer = None
