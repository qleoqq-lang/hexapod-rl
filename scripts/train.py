"""Convenience entrypoint; reuse the original project's PPO implementation."""
from _bootstrap import ROOT
from train_ppo import main
if __name__ == "__main__":
    main()
