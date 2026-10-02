"""HITL interrupt policy: which tools require approval on attended agents."""


def test_hitl_interrupt_on_arms_expected_tools():
    from EvoScientist.EvoScientist import HITL_INTERRUPT_ON

    assert HITL_INTERRUPT_ON == {
        "execute": True,
        "run_in_background": True,
        "schedule_task": True,
        "delete": True,
    }
