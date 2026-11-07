"""Simulation-only extraction of the archive's position-offset MIT PD loop."""
from pathlib import Path
import mujoco
import numpy as np
import yaml
from robot_interface import (LEG_ORDER, MJCF_PREFIX, JOINT_NAMES,
                             COMMAND_TO_MJCF_SIGNS, RobotState, make_observation)

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    with Path(path or ROOT / "configs/robot.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)

class SimRobot:
    def __init__(self, config):
        self.config = config
        path = Path(config["model"])
        self.model_path = path if path.is_absolute() else ROOT / path
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = JOINT_NAMES.copy()
        self.joint_ids = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.actuator_ids = np.array([self.model.actuator(n + "_motor").id for n in JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.base_id = self.model.body("base_link").id
        self.ground_id = self.model.geom("ground").id
        self.tibia_ids = [self.model.body(MJCF_PREFIX[l] + "_tibia_link").id for l in LEG_ORDER]
        if (self.model.nq, self.model.nv, self.model.nu) != (25, 24, 18):
            raise ValueError("Expected free base + 18 hinge joints + 18 motors")
        if not np.all(self.model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Expected hinge joints")
        if not np.array_equal(self.model.actuator_trnid[self.actuator_ids, 0], self.joint_ids):
            raise ValueError("Incorrect joint/actuator mapping")
        if not np.allclose(self.model.actuator_gear[self.actuator_ids, 0], 1):
            raise ValueError("PD torque assumes unit actuator gear")
        self.signs = np.array([s for l in LEG_ORDER for s in COMMAND_TO_MJCF_SIGNS[l]])
        self.stand_key = self.model.key("stand").id
        self.nominal_command = self.model.key_qpos[self.stand_key, self.qadr] * self.signs
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        self.kp = np.tile(config["kp"], 6)
        self.kd = np.tile(config["kd"], 6)
        self.frame_skip = int(config["frame_skip"])
        self.action_scale = float(config["action_scale_rad"])
        self.torque_limit = float(config["torque_limit_nm"])
        if self.frame_skip < 1 or self.action_scale <= 0 or self.torque_limit <= 0:
            raise ValueError("Invalid control configuration")
        if self.kp.shape != (18,) or self.kd.shape != (18,) or np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("kp/kd must contain three nonnegative gains")
        if np.any(self.torque_limit > self.model.actuator_ctrlrange[self.actuator_ids, 1]):
            raise ValueError("Configured torque exceeds MJCF actuator limits")
        self.dt = self.model.opt.timestep * self.frame_skip
        self.reset(np.random.default_rng(0), noise=False)

    def reset(self, rng, noise=True):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key)
        if noise:
            cfg = self.config["reset"]
            self.data.qpos[:2] += rng.uniform(-cfg["xy_noise_m"], cfg["xy_noise_m"], 2)
            self.data.qpos[self.qadr] += rng.uniform(-cfg["joint_noise_rad"], cfg["joint_noise_rad"], 18) * self.signs
            self.data.qvel[:] = rng.normal(0, cfg["velocity_noise"], self.model.nv)
        self.data.qpos[self.qadr] = np.clip(self.data.qpos[self.qadr], self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.assert_healthy()

    def assert_healthy(self):
        if not all(np.isfinite(a).all() for a in (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl)):
            raise FloatingPointError("Non-finite simulation state; training stopped")
        if any(w.number for w in self.data.warning):
            raise FloatingPointError("MuJoCo warning (including automatic instability reset); training stopped")

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite vector of shape (18,)")
        action = np.clip(action, -1, 1)
        target = np.clip((self.nominal_command + self.action_scale * action) * self.signs,
                         self.limits[:, 0], self.limits[:, 1])
        applied_action = (target * self.signs - self.nominal_command) / self.action_scale
        energy = 0.
        peak_torque = 0.
        max_violation = 0.
        for _ in range(self.frame_skip):
            q, v = self.data.qpos[self.qadr], self.data.qvel[self.vadr]
            torque = np.clip(self.kp * (target - q) - self.kd * v, -self.torque_limit, self.torque_limit)
            self.data.ctrl[self.actuator_ids] = torque
            energy += float(np.sum(np.abs(torque * v))) * self.model.opt.timestep
            peak_torque = max(peak_torque, float(np.max(np.abs(torque))))
            mujoco.mj_step(self.model, self.data)
            self.assert_healthy()
            max_violation = max(max_violation, self.joint_violation())
        mujoco.mj_forward(self.model, self.data)
        self.assert_healthy()
        return dict(energy=energy, peak_torque=peak_torque,
                    joint_violation_rad=max_violation, applied_action=applied_action)

    def joint_violation(self):
        q = self.data.qpos[self.qadr]
        return float(np.maximum(np.maximum(self.limits[:, 0] - q, q - self.limits[:, 1]), 0).max())

    def get_state(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        vel = np.zeros(6)
        # API order rot:lin. mjOBJ_BODY's local frame follows the inertial axes,
        # which are rotated in THIS model. Request world orientation, then rotate
        # explicitly into base_link axes (xmat), not the inertial frame (ximat).
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, vel, 0)
        contacts = np.zeros(6)
        for c in self.data.contact:
            if c.dist > 0 or self.ground_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.ground_id else c.geom1
            body = self.model.geom_bodyid[other]
            if body in self.tibia_ids:
                contacts[self.tibia_ids.index(body)] = 1
        return RobotState(self.data.xpos[self.base_id].copy(), self.data.xquat[self.base_id].copy(),
                          rotation.T @ vel[3:], rotation.T @ vel[:3], vel[3:].copy(),
                          rotation.T @ np.array([0., 0., -1.]),
                          self.data.qpos[self.qadr] * self.signs,
                          self.data.qvel[self.vadr] * self.signs, contacts)

    def get_observation(self, previous_action):
        return make_observation(self.get_state(), self.nominal_command, previous_action)

    def close(self):
        pass  # No device handles, sockets, or background control threads.
