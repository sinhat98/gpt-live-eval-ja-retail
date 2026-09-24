import pytest

from voice_eval.retail.data import ROOT, task
from voice_eval.retail.executor import RetailExecutor
from voice_eval.retail.identity import NAME_READINGS, READING_ONLY_MATCH, identities, identity_matches
from voice_eval.retail.phone_sequence import NEW_PHONE, TASK_USERS


def lookup(schemas):
    return next(s["function"] for s in schemas if s["function"]["name"] == "find_user_id_by_name_phone")


@pytest.mark.parametrize("uid", list(NAME_READINGS))
def test_reading_only_rejects_registered_kanji(uid):
    row = identities()[uid]
    reading = NAME_READINGS[uid]
    assert identity_matches(uid, row, **{**row, **reading}, reading_only=True)
    assert not identity_matches(uid, row, **row, reading_only=True)
    assert not identity_matches(uid, row, **{**row, **reading, "first_name": row["first_name"]}, reading_only=True)
    assert not identity_matches(uid, row, **{**row, **reading, "phone_number": "08000000000"}, reading_only=True)
    assert not identity_matches(uid, row, **{**row, **reading, "last_name": ""}, reading_only=True)
    assert not identity_matches("unregistered", row, **{**row, **reading}, reading_only=True)


@pytest.mark.parametrize("tid", TASK_USERS)
def test_name_kana_worker_matches_readings_only(tid):
    uid = TASK_USERS[tid]
    row = identities()[uid]
    state = {"task_id": tid, "phone_profile": f"sequence-{tid}-v1", "name_match": READING_ONLY_MATCH}
    with RetailExecutor(initial_state=state, task=task(tid) if tid == "control" else None) as e:
        assert "ひらがな" in lookup(e.schemas)["description"]
        kanji = {"first_name": row["first_name"], "last_name": row["last_name"], "phone_number": NEW_PHONE}
        assert not e.execute("find_user_id_by_name_phone", kanji, call_id="kanji")["ok"]
        katakana = {key: "".join(chr(ord(c) + 0x60) for c in value) for key, value in NAME_READINGS[uid].items()}
        result = e.execute("find_user_id_by_name_phone", {**katakana, "phone_number": NEW_PHONE}, call_id="kana")
        assert result == {"ok": True, "result": uid}
        assert e.evaluate()["replay_matches_snapshot"]


def test_default_mode_keeps_lookup_schema_and_kanji_match():
    with RetailExecutor(initial_state={"task_id": "5", "phone_profile": "sequence-5-v1"}) as e:
        assert "登録表記または" in lookup(e.schemas)["description"]
        args = {"first_name": "加奈", "last_name": "田中", "phone_number": NEW_PHONE}
        assert e.execute("find_user_id_by_name_phone", args, call_id="kanji")["ok"]


def test_name_kana_prompt_adds_only_reading_rule():
    base = ROOT / "prompts/retail/phone-sequence"
    variant = ROOT / "prompts/retail/name-kana"
    assert (base / "frontend.txt").read_bytes() == (variant / "frontend.txt").read_bytes()
    added = set((variant / "backend.txt").read_text().splitlines()) - set(
        (base / "backend.txt").read_text().splitlines()
    )
    assert added == {
        "氏名は漢字を推測せず、聞いた読みをひらがなで照合ツールに渡す。",
        (
            "氏名の読みを一文字ずつ確認しない（「さくら」の「さ」のような確認は求めない）。"
            "照合できないときは、氏名の読みをもう一度そのまま言ってもらう。"
        ),
    }
