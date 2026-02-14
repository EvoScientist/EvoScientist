"""Middleware package for EvoScientist.

Re-exports middleware classes and factory functions so that existing
``from EvoScientist.middleware import X`` imports continue to work.
"""

from deepagents.middleware.skills import SkillsMiddleware

from .memory import (
    EvoMemoryMiddleware,
    EvoMemoryState,
    ExtractedMemory,
    create_memory_middleware,
)


def create_skills_middleware(composite_backend) -> SkillsMiddleware:
    """Create a SkillsMiddleware that loads skills.

    Uses the CompositeBackend directly so that skill paths in the system
    prompt match the ``/skills/`` route (e.g. ``/skills/find-skills/SKILL.md``).

    Args:
        composite_backend: The CompositeBackend that routes ``/skills/`` to
            the MergedReadOnlyBackend.

    Returns:
        Configured SkillsMiddleware instance
    """
    return SkillsMiddleware(
        backend=composite_backend,
        sources=["/skills/"],
    )


__all__ = [
    "EvoMemoryMiddleware",
    "EvoMemoryState",
    "ExtractedMemory",
    "create_memory_middleware",
    "create_skills_middleware",
]
