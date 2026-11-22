"""TwinWorld P0: minimal, transparent CPU-only 1D paired-physics simulation.

Two visually identical bodies A and B move with same passive trajectory.
Hidden mechanism: coupled (invisible damped spring) or independent.
Only intervention on A can distinguish the worlds.

This is a SANITY CHECK benchmark, NOT an autonomous concept-learning method.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DT = 0.02
STEPS = 120
PULSE_STEPS = 18

@dataclass(frozen=True)
class Params:
    mass_a: float
    mass_b: float
    spring_k: float
    spring_d: float
    gamma: float
    distance: float
    velocity: float
    force: float


def sample_params(rng: np.random.Generator) -> Params:
    return Params(
        mass_a=float(rng.uniform(.7, 1.6)),
        mass_b=float(rng.uniform(.7, 1.6)),
        spring_k=float(rng.uniform(5.0, 17.0)),
        spring_d=float(rng.uniform(.3, 1.6)),
        gamma=float(rng.uniform(.04, .2)),
        distance=float(rng.uniform(.8, 1.5)),
        velocity=float(rng.uniform(-.45, .45)),
        force=float(rng.choice([-1., 1.]) * rng.uniform(3., 7.)),
    )


def simulate(p: Params, coupled: bool, force: bool, steps: int = STEPS) -> tuple[np.ndarray, np.ndarray]:
    # x[0] is A, x[1] is B; spring is hidden from observation.
    x = np.array([0.0, p.distance], dtype=np.float64)
    v = np.array([p.velocity, p.velocity], dtype=np.float64)
    positions = np.empty((steps, 2), dtype=np.float64)
    actions = np.zeros((steps,), dtype=np.float64)
    for t in range(steps):
        positions[t] = x
        f = p.force if (force and t < PULSE_STEPS) else 0.0
        actions[t] = f
        # Positive displacement of B relative to A pulls A right and B left.
        displacement = x[1] - x[0] - p.distance
        relative_speed = v[1] - v[0]
        # Remove only machine-rounding jitter in theoretically unstrained motion.
        if abs(displacement) < 1e-12 and abs(relative_speed) < 1e-12:
            displacement, relative_speed = 0.0, 0.0
        coupling = (p.spring_k * displacement + p.spring_d * relative_speed) if coupled else 0.0
        acc = np.array([
            (f + coupling) / p.mass_a - p.gamma * v[0],
            -coupling / p.mass_b - p.gamma * v[1],
        ])
        v = v + DT * acc
        x = x + DT * v
    return positions, actions
