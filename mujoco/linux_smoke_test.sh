#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

python3 "${SCRIPT_DIR}/hexapod_rl_env.py" \
  --model "${SCRIPT_DIR}/hexapod_rl.xml" \
  --steps 500

echo "Linux MuJoCo/RL smoke test completed."
