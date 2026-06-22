"""Gait pattern generation for hexapod robot.

Implements three standard gaits:
- Tripod gait (fast, 3 legs in stance at a time)
- Ripple gait (medium, 2-3 legs in stance)
- Wave gait (slow but stable, 1 leg lifted at a time)
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class GaitPhase:
    """Phase state for a single leg in a gait cycle."""
    leg_id: int
    phase_offset: float  # Phase offset within cycle [0, 1)
    is_stance: bool = True  # True = stance, False = swing
    swing_progress: float = 0.0  # Progress within swing phase [0, 1]


class GaitPlanner:
    """Generates foot trajectory targets for different gaits."""

    # Leg ordering: front-left(0), middle-left(1), hind-left(2),
    #               front-right(3), middle-right(4), hind-right(5)
    
    # Tripod gait: legs {0, 2, 4} swing together, {1, 3, 5} stance (and vice versa)
    TRIPOD_GROUPS = [
        [0, 2, 4],  # Group A
        [1, 3, 5],  # Group B
    ]
    
    # Ripple gait: each leg lifts in sequence with some overlap
    RIPPLE_PHASES = [0.0, 0.33, 0.67, 0.17, 0.50, 0.83]
    
    # Wave gait: legs move one by one in order
    WAVE_PHASES = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625]

    def __init__(
        self,
        cycle_time: float = 0.4,
        stance_ratio: float = 0.5,
        step_height: float = 0.05,
        stride_length: float = 0.06,
    ):
        self.cycle_time = cycle_time
        self.stance_ratio = stance_ratio
        self.swing_ratio = 1.0 - stance_ratio
        self.step_height = step_height
        self.stride_length = stride_length

    def get_gait_phases(
        self, gait_type: str, t: float
    ) -> List[GaitPhase]:
        """Get phase state for each leg at time t."""
        if gait_type == "tripod":
            phases = self._tripod_phases(t)
        elif gait_type == "ripple":
            phases = self._ripple_phases(t)
        elif gait_type == "wave":
            phases = self._wave_phases(t)
        else:
            phases = self._tripod_phases(t)
        return phases

    def _tripod_phases(self, t: float) -> List[GaitPhase]:
        """Tripod gait: groups A and B alternate."""
        cycle_t = t % self.cycle_time
        normalized_t = cycle_t / self.cycle_time  # [0, 1)
        
        phases = []
        for leg_id in range(6):
            in_group_a = leg_id in self.TRIPOD_GROUPS[0]
            # Group A swings in first half, Group B swings in second half
            if in_group_a:
                is_stance = normalized_t >= self.stance_ratio
                swing_prog = normalized_t / self.stance_ratio if normalized_t < self.stance_ratio else 0.0
            else:
                is_stance = normalized_t < self.stance_ratio
                swing_prog = (normalized_t - self.stance_ratio) / self.swing_ratio if normalized_t >= self.stance_ratio else 0.0
            
            phases.append(GaitPhase(
                leg_id=leg_id,
                phase_offset=0.0 if in_group_a else self.stance_ratio,
                is_stance=is_stance,
                swing_progress=min(swing_prog, 1.0) if not is_stance else 0.0,
            ))
        return phases

    def _ripple_phases(self, t: float) -> List[GaitPhase]:
        """Ripple gait: each leg has unique phase offset."""
        cycle_t = t % self.cycle_time
        normalized_t = cycle_t / self.cycle_time
        
        phases = []
        for leg_id in range(6):
            leg_phase = (normalized_t - self.RIPPLE_PHASES[leg_id]) % 1.0
            is_stance = leg_phase >= self.swing_ratio
            swing_prog = leg_phase / self.swing_ratio if leg_phase < self.swing_ratio else 0.0
            
            phases.append(GaitPhase(
                leg_id=leg_id,
                phase_offset=self.RIPPLE_PHASES[leg_id],
                is_stance=is_stance,
                swing_progress=min(swing_prog, 1.0) if not is_stance else 0.0,
            ))
        return phases

    def _wave_phases(self, t: float) -> List[GaitPhase]:
        """Wave gait: one leg at a time (most stable)."""
        cycle_t = t % self.cycle_time
        normalized_t = cycle_t / self.cycle_time
        
        phases = []
        for leg_id in range(6):
            leg_phase = (normalized_t - self.WAVE_PHASES[leg_id]) % 1.0
            is_stance = leg_phase >= self.swing_ratio
            swing_prog = leg_phase / self.swing_ratio if leg_phase < self.swing_ratio else 0.0
            
            phases.append(GaitPhase(
                leg_id=leg_id,
                phase_offset=self.WAVE_PHASES[leg_id],
                is_stance=is_stance,
                swing_progress=min(swing_prog, 1.0) if not is_stance else 0.0,
            ))
        return phases

    def get_foot_offsets(
        self, gait_type: str, t: float, leg_default_positions: np.ndarray
    ) -> np.ndarray:
        """Compute foot position offsets for each leg at time t.

        Args:
            gait_type: One of 'tripod', 'ripple', 'wave'
            t: Current time in seconds
            leg_default_positions: (6, 3) array of default foot positions

        Returns:
            (6, 3) array of target foot positions
        """
        phases = self.get_gait_phases(gait_type, t)
        targets = leg_default_positions.copy()

        for i, phase in enumerate(phases):
            if phase.is_stance:
                # Stance phase: foot moves backward relative to body
                stance_progress = 1.0 - (t % self.cycle_time) / \
                    (self.cycle_time * self.stance_ratio)
                # Keep foot on ground, move backward
                targets[i, 0] -= self.stride_length * (1.0 - stance_progress)
            else:
                # Swing phase: foot lifts and moves forward
                # Elliptical trajectory
                swing_x = -self.stride_length * (0.5 - phase.swing_progress)
                swing_z = self.step_height * np.sin(
                    np.pi * phase.swing_progress
                )
                targets[i, 0] += swing_x
                targets[i, 2] += swing_z

        return targets
