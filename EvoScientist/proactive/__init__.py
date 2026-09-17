"""Proactive turns (issue #263).

Home of the proactive-shadow machinery: a proactive turn reasons over a real,
idle chat's history and decides whether to push an unprompted message, without
persisting a synthetic trigger into the real thread and without executing any
tool against the real sandbox.
"""
