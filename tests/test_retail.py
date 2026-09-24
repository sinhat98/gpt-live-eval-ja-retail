import pytest

from voice_eval.retail.data import PILOT, task
from voice_eval.retail.executor import RetailExecutor


def test_schema_complete_and_idempotence():
    with RetailExecutor(task_id="21") as e:
        assert len(e.schemas) == 15
        assert len({s["function"]["name"] for s in e.schemas}) == 15
        assert e.execute("calculate", {"expression": "1+2"}, call_id="a")["result"] == "3.0"
        assert e.execute("calculate", {"expression": "1+2"}, call_id="a")["result"] == "3.0"
        assert len(e.executions) == 1
        with pytest.raises(ValueError):
            e.execute("calculate", {"expression": "1+3"}, call_id="a")
        assert e.execute("get_order_details", {"order_id": "invalid"}, call_id="bad")["ok"] is False


@pytest.mark.parametrize("tid", PILOT)
def test_gold_replay_and_snapshot(tid):
    t = task(tid)
    with RetailExecutor(task_id=tid, task=t if tid == "control" else None) as e:
        for a in t["evaluation_criteria"]["actions"]:
            assert e.execute(a["name"], a["arguments"], call_id=a["action_id"])["ok"]
        grade = e.evaluate()
        assert grade["environment"]["db_check"]["db_match"]
        assert grade["replay_matches_snapshot"]


def test_trial_isolation_and_worker_crash():
    with RetailExecutor(task_id="22") as a, RetailExecutor(task_id="22") as b:
        args = task("22")["evaluation_criteria"]["actions"][1]["arguments"]
        a.execute("modify_user_address", args, call_id="write")
        assert a.snapshot() != b.snapshot()
        a.process.kill()
        a.process.wait()
        with pytest.raises(RuntimeError):
            a.snapshot()


def test_all_remaining_operations_and_rejections():
    # Derive legal variants from the pinned database, without hardcoding prices.
    with RetailExecutor(task_id="21") as e:
        db = e.snapshot()
        assert e.execute("list_all_product_types", {}, call_id="list")["ok"]
        order = next(
            o
            for o in db["orders"].values()
            if o["status"] == "保留中"
            and any(
                not k.startswith("gift_card") and k != o["payment_history"][0]["payment_method_id"]
                for k in db["users"][o["user_id"]]["payment_methods"]
            )
        )
        uid = order["user_id"]
        user = db["users"][uid]
        payments = user["payment_methods"]
        alternate = next(
            k
            for k in payments
            if not k.startswith("gift_card") and k != order["payment_history"][0]["payment_method_id"]
        )
        result = e.execute(
            "modify_pending_order_payment",
            {"order_id": order["order_id"], "payment_method_id": alternate},
            call_id="payment",
        )
        assert result["ok"]
        assert any(
            p["payment_method_id"] == alternate and p["transaction_type"] == "payment"
            for p in result["result"]["payment_history"]
        )
        assert not e.execute(
            "cancel_pending_order", {"order_id": order["order_id"], "reason": "INVALID"}, call_id="bad_reason"
        )["ok"]
        cancelled = e.execute(
            "cancel_pending_order", {"order_id": order["order_id"], "reason": "不要になった"}, call_id="cancel"
        )
        assert cancelled["result"]["status"] == "キャンセル済み"
        assert cancelled["result"]["payment_history"][-1]["transaction_type"] == "refund"
        assert not e.execute(
            "cancel_pending_order", {"order_id": order["order_id"], "reason": "不要になった"}, call_id="cancel_again"
        )["ok"]
    with RetailExecutor(task_id="5") as e:
        db = e.snapshot()
        order = db["orders"]["#W6390527"]
        old = order["items"][0]
        product = db["products"][old["product_id"]]
        variant = next(v for v in product["variants"].values() if v["available"] and v["item_id"] != old["item_id"])
        params = {
            "order_id": order["order_id"],
            "item_ids": [old["item_id"]],
            "new_item_ids": [variant["item_id"]],
            "payment_method_id": order["payment_history"][0]["payment_method_id"],
        }
        exchanged = e.execute("exchange_delivered_order_items", params, call_id="exchange")
        assert exchanged["ok"]
        assert exchanged["result"]["exchange_price_difference"] == round(variant["price"] - old["price"], 2)
        assert not e.execute("exchange_delivered_order_items", params, call_id="exchange_again")["ok"]
        assert not e.execute(
            "cancel_pending_order",
            {"order_id": order["order_id"], "reason": "不要になった"},
            call_id="cancel_delivered",
        )["ok"]


def test_variant_stock_balance_and_once_only():
    with RetailExecutor(task_id="21") as e:
        args = task("21")["evaluation_criteria"]["actions"][-1]["arguments"]
        wrong = dict(args, new_item_ids=["1421289881", "4107812777"])
        assert not e.execute("modify_pending_order_items", wrong, call_id="cross_product")["ok"]
        assert not e.execute(
            "modify_pending_order_payment",
            {"order_id": "#W9911714", "payment_method_id": "gift_card_4332117"},
            call_id="balance",
        )["ok"]
        result = e.execute("modify_pending_order_items", args, call_id="items")
        assert result["ok"]
        assert e.snapshot()["users"]["akemi_sasaki_1261"]["payment_methods"]["gift_card_4332117"]["balance"] == 6612
        assert not e.execute("modify_pending_order_items", args, call_id="again")["ok"]
        assert not e.execute(
            "cancel_pending_order", {"order_id": "#W9911714", "reason": "不要になった"}, call_id="cancel"
        )["ok"]


def test_refund_cross_order_rejected_and_two_original_refunds():
    t = task("11")
    with RetailExecutor(task_id="11") as e:
        actions = [a for a in t["evaluation_criteria"]["actions"] if a["name"] == "return_delivered_order_items"]
        wrong = dict(actions[0]["arguments"], payment_method_id=actions[1]["arguments"]["payment_method_id"])
        assert not e.execute("return_delivered_order_items", wrong, call_id="wrong")["ok"]
        for i, a in enumerate(actions):
            assert e.execute(a["name"], a["arguments"], call_id=str(i))["ok"]
        assert e.evaluate()["environment"]["db_check"]["db_match"]


def test_checkpoint_replay_has_no_future_actions():
    import json

    from voice_eval.retail.data import DATA

    for tid in ("21", "22"):
        fixture = json.loads((DATA / "checkpoints" / f"{tid}.json").read_text())
        with RetailExecutor(initial_state={"task_id": tid, "checkpoint_calls": fixture["calls"]}) as e:
            assert e.snapshot() == fixture["snapshot"]
            if tid == "21":
                assert not any(c["name"].startswith("modify") for c in e.executions)
            else:
                assert e.snapshot()["users"]["akemi_sasaki_1261"]["address"]["zip"] == "100-0001"
                assert e.snapshot()["orders"]["#W9911714"]["address"]["zip"] == "100-0001"


def test_bad_traces_fail_independent_audit():
    from voice_eval.retail.audit import audit_calls

    with RetailExecutor(task_id="21") as e:
        initial = e.snapshot()

    def event(name, args, result=True):
        return {"name": name, "arguments": args, "call_id": name, "output": {"ok": True, "result": result}}

    lookup = event("find_user_id_by_name_phone", {}, "akemi_sasaki_1261")
    assert not audit_calls(initial, [event("get_user_details", {"user_id": "akemi_sasaki_1261"})], task_id="21")[
        "passed"
    ]
    assert not audit_calls(initial, [lookup, event("get_user_details", {"user_id": "kana_tanaka_8020"})], task_id="5")[
        "passed"
    ]
    assert not audit_calls(
        initial, [lookup, event("modify_pending_order_items", {"order_id": "#W9911714"})], task_id="21", partial=True
    )["passed"]
    assert not audit_calls(
        initial, [lookup, event("modify_pending_order_address", {"order_id": "#W9911714"})], task_id="22", partial=True
    )["passed"]
    assert not audit_calls(initial, [], task_id="10")["passed"]


@pytest.mark.parametrize("tid", ["5", "11", "22"])
def test_known_incorrect_terminal_states_do_not_match_gold(tid):
    t = task(tid)
    with RetailExecutor(task_id=tid) as e:
        if tid == "11":
            a = next(a for a in t["evaluation_criteria"]["actions"] if a["name"] == "return_delivered_order_items")
            assert e.execute(a["name"], a["arguments"], call_id="only_one_order")["ok"]
        elif tid == "22":
            for a in t["evaluation_criteria"]["actions"]:
                assert e.execute(a["name"], a["arguments"], call_id=a["action_id"])["ok"]
            args = dict(t["evaluation_criteria"]["actions"][-1]["arguments"])
            args.pop("user_id")
            args["order_id"] = "#W9911714"
            assert e.execute("modify_pending_order_address", args, call_id="wrong_scope")["ok"]
        else:
            # Never act on the latest return request, leaving the task unfinished.
            assert e.execute("get_order_details", {"order_id": "#W6390527"}, call_id="read_only")["ok"]
        assert not e.evaluate()["environment"]["db_check"]["db_match"]


def test_gemini_cache_includes_instruction(tmp_path, monkeypatch):
    from crawl_harness.audio_cache import CallerAudioCache
    from shared.audio import gemini

    cache = CallerAudioCache(tmp_path)
    params = {"text": "日本語", "model": "gemini-test", "voice": "Kore", "sample_rate_hz": 24000}
    before = cache.path_for("test", **params)
    monkeypatch.setattr(gemini, "INSTRUCTION", "changed instruction")
    assert cache.path_for("test", **params) != before


def test_all_source_tasks_load_and_gold_error_fails_preflight():
    from voice_eval.retail.data import tasks

    source = tasks()
    assert len(source) > len(PILOT)
    nonpilot = next(k for k in source if k not in PILOT)
    with RetailExecutor(task_id=nonpilot) as e:
        assert len(e.schemas) == 15
    t = task("control")
    t["evaluation_criteria"]["actions"][0]["arguments"]["order_id"] = "missing"
    with RetailExecutor(task=t) as e, pytest.raises(ValueError):
        e.evaluate()


def test_unavailable_variant_rejected_in_isolated_fixture():
    t = task("21")
    with RetailExecutor(task_id="21") as e:
        db = e.snapshot()
    db["products"]["6938111410"]["variants"]["4107812777"]["available"] = False
    t["initial_state"] = {
        "initialization_data": {"agent_data": db},
        "initialization_actions": None,
        "message_history": [],
    }
    with RetailExecutor(task=t) as e:
        args = t["evaluation_criteria"]["actions"][-1]["arguments"]
        assert not e.execute("modify_pending_order_items", args, call_id="stock")["ok"]
        assert e.snapshot() == e.initial_snapshot


def test_completion_projection_avoids_full_database_rpc(monkeypatch):
    with RetailExecutor(task_id="22") as e:
        uid = "akemi_sasaki_1261"
        e.execute("get_user_details", {"user_id": uid}, call_id="user")
        action = task("22")["evaluation_criteria"]["actions"][1]
        e.execute(action["name"], action["arguments"], call_id="write")

        def no_rpc(*args, **kwargs):
            raise AssertionError("Audio completion loop must not fetch the full DB")

        with monkeypatch.context() as m:
            m.setattr(e, "rpc", no_rpc)
            before, after = e.completion_evidence_state()
            assert set(before["users"]) == {uid}
            assert before["users"][uid]["address"]["zip"] == "689-0831"
            assert after["users"][uid]["address"]["zip"] == "100-0001"
            after["users"][uid]["address"]["zip"] = "tampered"
            assert e.completion_evidence_state()[1]["users"][uid]["address"]["zip"] == "100-0001"


def test_empty_nl_assertions_are_not_a_failure():
    from voice_eval.retail.scoring import finalize

    r = finalize(
        {
            "partial": False,
            "environment": {"reward_breakdown": {"DB": 1.0}},
            "reward_basis": ["DB", "NL_ASSERTION"],
            "nl_assertions": [],
            "semantic": {"policy": "pass", "caller_validity": "valid", "nl_assertions": "unknown"},
            "deterministic_audit": {"passed": True},
            "handoff_passed": True,
            "replay_matches_snapshot": True,
        }
    )
    assert r["task_reward"] == 1.0 and r["passed"]


def test_final_user_consent_is_delivered_before_end(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import openai

    import voice_eval.retail.text as module

    requests = []

    class Client:
        responses = None

        def __init__(self, **kwargs):
            self.responses = self

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                output_text="はい、その内容で進めてください。<END>" if len(requests) == 1 else "受け付けました。",
                output=[],
                usage=SimpleNamespace(model_dump=dict),
            )

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(module, "score", lambda *args, **kwargs: {"status": "passed", "passed": True})
    module.run_text("control", tmp_path)
    assert len(requests) == 2
    assert requests[1]["input"][0]["content"] == "はい、その内容で進めてください。"


def test_source_blob_hashes_match_fixed_commit():
    import hashlib
    import json

    from voice_eval.retail.data import ROOT

    root = ROOT / "vendor/j_tau"
    manifest = json.loads((root / "SOURCE_FILES.json").read_text())
    for path, sha in manifest["blobs"].items():
        raw = (root / path).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == sha


def test_rejected_forbidden_write_is_still_a_policy_failure():
    from voice_eval.retail.audit import audit_calls

    events = [
        {
            "name": "return_delivered_order_items",
            "arguments": {"order_id": "x"},
            "call_id": "bad",
            "output": {"ok": False, "error": "Payment method should be the original payment method"},
        }
    ]
    result = audit_calls({"orders": {}}, events, task_id="11")
    assert any(v["check"] == "business_rule_rejection" for v in result["violations"])


def test_candidate_preserves_backend_and_has_no_task_answers():
    from voice_eval.retail.data import ROOT

    prompts = ROOT / "prompts/retail"
    assert (prompts / "baseline/backend.txt").read_bytes() == (prompts / "candidate/backend.txt").read_bytes()
    frontend = (prompts / "candidate/frontend.txt").read_text()
    assert not any(x in frontend for x in ("佐々木", "田中", "長谷川", "4107812777", "6612"))


def test_caller_review_cannot_see_internal_results():
    from voice_eval.retail.caller_audit import public_dialogue

    assert (
        public_dialogue("USER 0ms: はい\nBACKEND 1ms: private\nTOOL 2ms: secret\nASSISTANT 3ms: どうぞ")
        == "USER 0ms: はい\nASSISTANT 3ms: どうぞ"
    )
    assert public_dialogue([{"role": "tool", "content": "secret"}, {"role": "user", "content": "はい"}]) == [
        {"role": "user", "content": "はい"}
    ]


def test_caller_fact_error_overrides_inconsistent_valid_label():
    import json
    from types import SimpleNamespace

    from voice_eval.retail.caller_audit import audit_caller

    payload = {
        "status": "valid",
        "known_fact_errors": [{"expected": "100-0001", "observed": "111-1111"}],
        "reason": "誤った番号に同意",
    }
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                output_text=json.dumps(payload), usage=SimpleNamespace(model_dump=dict)
            )
        )
    )
    assert audit_caller(task("22"), "USER 0ms: はい", client)["status"] == "invalid"


def test_input_audio_invalid_is_detected_before_inference(tmp_path):
    import wave

    from voice_eval.retail.audio_review import validate_input_wav

    p = tmp_path / "bad.wav"
    p.write_bytes(b"not audio")
    with pytest.raises(ValueError):
        validate_input_wav(p)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\0" * 48000)
    with pytest.raises(ValueError, match="silent"):
        validate_input_wav(p)


def test_interrupted_result_write_preserves_previous_record(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from voice_eval.retail.data import save

    p = tmp_path / "result.json"
    save(p, {"status": "passed"})
    original = Path.write_text

    def broken_write(path, *args, **kwargs):
        if path.suffix == ".tmp":
            original(path, "partial")
            raise OSError("simulated interruption")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", broken_write)
    with pytest.raises(OSError):
        save(p, {"status": "new"})
    assert json.loads(p.read_text()) == {"status": "passed"}
    assert not list(tmp_path.glob("*.tmp"))


def test_phone_identity_requires_both_fields_and_hides_legacy_lookup():
    from voice_eval.retail.identity import identities

    row = identities()["akemi_sasaki_1261"]
    with RetailExecutor(task_id="21") as e:
        names = {s["function"]["name"] for s in e.schemas}
        assert "find_user_id_by_name_phone" in names
        assert "find_user_id_by_email" not in names
        assert "find_user_id_by_name_zip" not in names
        assert e.execute("find_user_id_by_name_phone", row, call_id="phone")["result"] == "akemi_sasaki_1261"
        normalized = {**row, "phone_number": row["phone_number"].replace("-", "")}
        assert e.execute("find_user_id_by_name_phone", normalized, call_id="normalized")["ok"]
        for index, args in enumerate(({**row, "phone_number": "000"}, {**row, "first_name": "別人"})):
            assert not e.execute("find_user_id_by_name_phone", args, call_id=f"bad-{index}")["ok"]
        assert not e.execute("find_user_id_by_email", {"email": "unused"}, call_id="old")["ok"]
        assert e.evaluate()["replay_matches_snapshot"]
    assert row["phone_number"] in task("21")["user_scenario"]["instructions"]["known_info"]
