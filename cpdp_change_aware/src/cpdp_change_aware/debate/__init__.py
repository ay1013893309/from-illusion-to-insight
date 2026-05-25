"""Debate pipeline."""

from .static_system import StaticTargetDebateSystem
from .system import CrossProjectDebateSystem

__all__ = ["CrossProjectDebateSystem", "StaticTargetDebateSystem"]
