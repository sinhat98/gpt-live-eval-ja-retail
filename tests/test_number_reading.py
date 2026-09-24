import copy
import json

import pytest

from voice_eval.retail.data import DATA
from voice_eval.retail.executor import RetailExecutor
from voice_eval.retail.number_reading import DIGITS, caller_projection, format_identifier


@pytest.mark.parametrize("value", ["050-0000-0102", "4107812777", "#W9911714", "100-0001", "０５０-００００-０００８"])
def test_reading_preserves_every_digit(value):
    r = format_identifier(value)
    spoken = r["spoken"].removeprefix("シャープ・ダブリュー・").removeprefix("ダブリュー・")
    recovered = "".join(str(DIGITS.index(word)) for word in spoken.replace("／", "・").split("・"))
    assert recovered == r["digits"]
    assert len(recovered) == r["digit_count"]


@pytest.mark.parametrize("value", ["050-??-0102", "050 / 0000", "6,612円", "0.5", ""])
def test_does_not_guess_invalid_input(value):
    with pytest.raises(ValueError):
        format_identifier(value)


def test_private_projection_does_not_mutate_business_or_expected_fields():
    scenario = json.loads((DATA / "all_run.json").read_text())["scenarios"][5]
    original = copy.deepcopy(scenario)
    scenario["simulation_parameters"]["known_facts"]["test"] = "電話は050-0000-0102。金額は1234567890円。個数は2個。"
    projected, readings = caller_projection(scenario)
    text = projected["simulation_parameters"]["known_facts"]["test"]
    assert "050-0000-0102" not in text
    assert "ゼロ・ゴ・ゼロ／ゼロ・ゼロ・ゼロ・ゼロ／ゼロ・イチ・ゼロ・ニ" in text
    assert "1234567890円" in text and "2個" in text
    assert projected["expected"] == original["expected"]
    assert projected["application"] == original["application"]
    assert scenario["simulation_parameters"]["known_facts"]["test"].startswith("電話は050")
    assert any(row["canonical"] == "050-0000-0102" for row in readings)


def test_reading_tool_is_opt_in_and_does_not_change_state(monkeypatch):
    monkeypatch.delenv("RETAIL_NUMBER_READING", raising=False)
    with RetailExecutor(task_id="5") as e:
        assert len(e.schemas) == 15
        assert not e.execute("format_spoken_identifier", {"value": "050-0000-0102"}, call_id="disabled")["ok"]
    monkeypatch.setenv("RETAIL_NUMBER_READING", "katakana-v1")
    with RetailExecutor(task_id="5") as e:
        assert len(e.schemas) == 16
        before = e.snapshot()
        r = e.execute("format_spoken_identifier", {"value": "050-0000-0102"}, call_id="reading")
        assert r["ok"] and r["result"]["digits"] == "05000000102"
        assert e.snapshot() == before
        assert e.evaluate()["replay_matches_snapshot"]
