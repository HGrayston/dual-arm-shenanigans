"""Dual-arm SO-101 simulation in MuJoCo."""

from .scene import (
    ARMS,
    JOINTS,
    READY_Q,
    TABLE_HEIGHT,
    ArmIds,
    arm_ids,
    build_spec,
    load,
)

__all__ = [
    "ARMS",
    "JOINTS",
    "READY_Q",
    "TABLE_HEIGHT",
    "ArmIds",
    "arm_ids",
    "build_spec",
    "load",
]
__version__ = "0.1.0"
