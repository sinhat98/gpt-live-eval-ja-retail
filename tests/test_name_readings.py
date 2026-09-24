import pytest

from voice_eval.retail.executor import RetailExecutor
from voice_eval.retail.identity import NAME_READINGS, identities, identity_matches


@pytest.mark.parametrize("uid", list(NAME_READINGS))
def test_registered_readings_require_name_and_phone(uid):
    row = identities()[uid]
    reading = NAME_READINGS[uid]
    assert identity_matches(uid, row, **row)
    assert identity_matches(uid, row, **{**row, **reading})
    assert not identity_matches(uid, row, **{**row, **reading, "phone_number": "08000000000"})
    assert not identity_matches(uid, row, **{**row, **reading, "first_name": ""})
    assert not identity_matches(uid, row, **{**row, **reading, "last_name": "別人"})


def test_kana_lookup_in_worker_and_replay():
    with RetailExecutor(task_id="5", initial_state={"phone_profile": "sequence-5-v1"}) as executor:
        for i, (first, last) in enumerate([("かな", "田中"), ("カナ", "タナカ"), ("ｶﾅ", "ﾀﾅｶ")]):
            result = executor.execute(
                "find_user_id_by_name_phone",
                {"first_name": first, "last_name": last, "phone_number": "080-1234-5678"},
                call_id=f"kana-{i}",
            )
            assert result == {"ok": True, "result": "kana_tanaka_8020"}
        assert executor.evaluate()["replay_matches_snapshot"]


def test_no_inferred_kanji_or_partial_or_unregistered_reading():
    uid = "kyosuke_hasegawa_4516"
    row = identities()[uid]
    for name in ["京介", "きょう", "きょーすけ"]:
        assert not identity_matches(uid, row, **{**row, "first_name": name})
    assert identity_matches(uid, row, **{**row, "first_name": "キョウスケ", "last_name": "ハセガワ"})
    assert not identity_matches("unregistered", row, **{**row, "first_name": "きょうすけ"})
