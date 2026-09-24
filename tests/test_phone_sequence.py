import pytest

from voice_eval.retail.data import ROOT, task
from voice_eval.retail.executor import RetailExecutor
from voice_eval.retail.identity import identities
from voice_eval.retail.phone_sequence import NEW_PHONE, TASK_USERS, project


@pytest.mark.parametrize("tid", TASK_USERS)
def test_profile_is_isolated_and_consistent_with_caller(tid):
    original = task(tid)
    uid = TASK_USERS[tid]
    old_phone = identities()[uid]["phone_number"]
    updated = project(original, tid)
    assert old_phone in str(original["user_scenario"])
    assert NEW_PHONE in str(updated["user_scenario"])
    assert updated["evaluation_criteria"]["nl_assertions"] == original["evaluation_criteria"]["nl_assertions"]
    name = {key: identities()[uid][key] for key in ("first_name", "last_name")}
    with (
        RetailExecutor(
            initial_state={"task_id": tid, "phone_profile": f"sequence-{tid}-v1"},
            task=updated if tid == "control" else None,
        ) as changed,
        RetailExecutor(task_id=tid, task=original if tid == "control" else None) as baseline,
    ):
        assert changed.execute("find_user_id_by_name_phone", {**name, "phone_number": NEW_PHONE}, call_id="new")["ok"]
        assert not changed.execute("find_user_id_by_name_phone", {**name, "phone_number": old_phone}, call_id="old")[
            "ok"
        ]
        assert baseline.execute("find_user_id_by_name_phone", {**name, "phone_number": old_phone}, call_id="old")["ok"]
        assert not baseline.execute("find_user_id_by_name_phone", {**name, "phone_number": NEW_PHONE}, call_id="new")[
            "ok"
        ]
        assert changed.evaluate()["replay_matches_snapshot"]
        assert changed.snapshot() == baseline.snapshot()
    assert identities()[uid]["phone_number"] == old_phone
    assert identities(f"sequence-{tid}-v1")[uid]["phone_number"] == NEW_PHONE


def test_number_only_variant_keeps_prompts_identical():
    for name in ("frontend.txt", "backend.txt"):
        assert (ROOT / "prompts/retail/baseline" / name).read_bytes() == (
            ROOT / "prompts/retail/phone-sequence" / name
        ).read_bytes()
