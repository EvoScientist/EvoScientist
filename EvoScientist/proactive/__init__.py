"""Proactive research pushes (issue #263).

Decide-core: reason over a real idle chat's history and decide whether to push
an unprompted assistant message, without persisting a synthetic trigger or
running tools against the real sandbox. The decision (gate -> tool-stripped
shadow turn -> commit decision) is pure and serve-agnostic; serve applies and
delivers the decided push in-process (Shape B), so this package holds no
out-of-band writer.
"""
