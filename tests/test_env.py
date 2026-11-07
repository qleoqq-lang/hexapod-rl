"""Boundary tests that random rollouts cannot guarantee to exercise."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mujoco"))
import mujoco
import numpy as np
from hexapod_rl_env import HexapodMuJoCoEnv

class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = HexapodMuJoCoEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=42, options={"noise": False})

    def test_seed_and_previous_action_reset(self):
        a, _ = self.env.reset(seed=17)
        self.env.step(np.ones(18) * .1)
        b, _ = self.env.reset(seed=17)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(b[-18:], np.zeros(18))

    def test_invalid_actions_do_not_advance_physics(self):
        for action in [np.zeros(17), np.full(18, np.nan), np.full(18, np.inf)]:
            before = self.env.data.time
            with self.assertRaises(ValueError):
                self.env.step(action)
            self.assertEqual(before, self.env.data.time)

    def test_action_clipping_and_observation_history(self):
        obs, _, _, _, _ = self.env.step(np.full(18, 5.))
        np.testing.assert_allclose(obs[-18:], np.ones(18))
        self.assertLessEqual(np.abs(self.env.data.ctrl).max(), 7.)

    def test_world_to_body_velocity(self):
        self.env.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        self.env.data.qvel[:] = 0
        self.env.data.qvel[0] = 1
        mujoco.mj_forward(self.env.model, self.env.data)
        state = self.env.robot.get_state()
        np.testing.assert_allclose(state.linear_velocity, [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(state.world_linear_velocity, [1, 0, 0], atol=1e-12)

    def test_low_high_and_flipped_termination(self):
        for z, quat, expected in [(0.05, [1, 0, 0, 0], "height_low"),
                                  (.5, [1, 0, 0, 0], "height_high"),
                                  (.2, [0, 1, 0, 0], "flipped"),
                                  (.2, [np.cos(.7), np.sin(.7), 0, 0], "excessive_tilt")]:
            self.env.reset(options={"noise": False})
            self.env.data.qpos[2] = z
            self.env.data.qpos[3:7] = quat
            mujoco.mj_forward(self.env.model, self.env.data)
            self.assertIn(expected, self.env.termination_reasons(self.env.robot.get_state()))

    def test_joint_limit_failure(self):
        r = self.env.robot
        self.env.data.qpos[r.qadr[0]] = r.limits[0, 1] + .03
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertIn("joint_limit", self.env.termination_reasons(r.get_state(), r.joint_violation()))

    def test_timeout_is_not_failure(self):
        self.env.step_count = self.env.max_steps - 1
        _, _, terminated, truncated, info = self.env.step(np.zeros(18))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reward_terms"]["termination"], 0.)
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(18))

    def test_failure_penalty_and_reset(self):
        self.env.data.qpos[2] = .5
        mujoco.mj_forward(self.env.model, self.env.data)
        _, _, terminated, _, info = self.env.step(np.zeros(18))
        self.assertTrue(terminated)
        self.assertEqual(info["reward_terms"]["termination"], -5.)
        self.env.reset()
        self.assertFalse(self.env.needs_reset)

if __name__ == "__main__":
    unittest.main()
