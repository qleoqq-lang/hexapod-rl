"""Six interpretable terms. All weights live in robot.yaml."""
import numpy as np

def compute_reward(state, action, previous_action, energy, terminated, config):
    upright = float(-state.projected_gravity[2])
    cap = config["forward_speed_cap_m_s"]
    terms = {
        "forward": config["forward"] * float(np.clip(state.world_linear_velocity[0], -cap, cap)),
        "orientation": -config["orientation"] * (1 - upright),
        "angular_velocity": -config["angular_velocity"] * float(np.mean(state.angular_velocity ** 2)),
        "action_rate": -config["action_rate"] * float(np.mean((action - previous_action) ** 2)),
        "energy": -config["energy"] * energy,
        "termination": -config["termination"] * float(terminated),
    }
    return float(sum(terms.values())), terms
