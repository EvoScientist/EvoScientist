"""Proactive pushes (#263).

A conversation armed with ``/proactive on`` may get one unprompted assistant
message after it has gone quiet. This package holds the decision: a
deterministic gate, a shadow turn that runs the real agent in memory with every
tool stripped, and the checks that turn the shadow's reply into a push or not.
It never writes to a user thread; serve applies a decided push.
"""
