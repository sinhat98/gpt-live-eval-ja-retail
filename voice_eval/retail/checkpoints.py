"""Legal, deterministic checkpoint histories; independent of future gold actions."""

from voice_eval.retail.data import DATA, save
from voice_eval.retail.executor import RetailExecutor


def create_checkpoint(tid):
    from voice_eval.retail.identity import identities

    phone = identities()["akemi_sasaki_1261"]["phone_number"]
    calls = []
    history = [
        {"role": "user", "text": f"佐々木明美です。登録電話番号は{phone}です。"},
    ]
    with RetailExecutor(task_id=tid) as e:

        def call(name, **arguments):
            item = {"name": name, "arguments": arguments, "call_id": f"checkpoint-{len(calls)}"}
            result = e.execute(**item)
            if not result["ok"]:
                raise ValueError(f"Invalid checkpoint call: {name}")
            calls.append(item)
            return result["result"]

        uid = call("find_user_id_by_name_phone", first_name="明美", last_name="佐々木", phone_number=phone)
        user = call("get_user_details", user_id=uid)
        orders = [call("get_order_details", order_id=oid) for oid in user["orders"]]
        history.append({"role": "assistant", "text": "本人確認ができました。ご注文内容を確認しました。"})
        if tid == "21":
            history.append({"role": "user", "text": "靴を4107812777に変更して、差額はギフトカードで払いたいです。"})
            item = call("get_item_details", item_id="4107812777")
            order = next(o for o in orders if o["order_id"] == "#W9911714")
            old = next(i for i in order["items"] if i["name"] == "ランニングシューズ")
            product = call("get_product_details", product_id=old["product_id"])
            if "4107812777" not in product["variants"]:
                raise ValueError("Checkpoint replacement must belong to the same product")
            difference = call("calculate", expression=f"{item['price']} - {old['price']}")
            history.append(
                {
                    "role": "assistant",
                    "text": f"保留中の注文#W9911714の靴を4107812777へ変更し、差額{difference}円を登録済みギフトカードで支払います。この内容でよろしいですか。",
                }
            )
            utterance = "その前に、アイテムID1656367028も1421289881へ変更してください。もしかすると最初の番号は製品IDかもしれません。両方まとめた内容と差額を教えてください。"
        else:
            address = {
                "address1": "丸の内1-2-3",
                "address2": "",
                "city": "千代田区",
                "state": "東京都",
                "country": "日本",
                "zip": "100-0001",
            }
            old = user["address"]
            history += [
                {
                    "role": "user",
                    "text": "登録住所とすべての注文の住所を、〒100-0001東京都千代田区丸の内1-2-3へ変更してください。",
                },
                {
                    "role": "assistant",
                    "text": "登録住所と保留中の注文#W9911714の配送先をその住所へ変更します。他の注文は保留中ではなく変更できません。この2件を変更してよろしいですか。",
                },
                {"role": "user", "text": "はい、その2件の住所を変更してください。"},
            ]
            call("modify_user_address", user_id=uid, **address)
            call("modify_pending_order_address", order_id="#W9911714", **address)
            history.append(
                {
                    "role": "assistant",
                    "text": f"登録住所と注文#W9911714の配送先を変更しました。元の登録住所は{old}でした。",
                }
            )
            utterance = "やっぱり登録住所だけ元の住所に戻したいです。注文の配送先は今変更したままにしてください。戻す内容を確認させてください。"
        snapshot = e.snapshot()
        save(
            DATA / "checkpoints" / f"{tid}.json",
            {"calls": calls, "history": history, "snapshot": snapshot, "executions": e.executions},
        )
        # Only information already obtained legitimately enters the summary.
        import json

        context = {
            "history": history,
            "summary": "この会話で実行済みのツール結果（将来の正解操作は含まない）:\n"
            + json.dumps(
                [{k: v for k, v in call.items() if k != "duration_ms"} for call in e.executions], ensure_ascii=False
            ),
        }
        scenario = {
            "id": f"retail-{tid}-checkpoint",
            "title": f"日本語retail {tid}の意向変更",
            "type": "retail",
            "interaction": "single_turn",
            "tags": ["ja", "retail", "checkpoint"],
            "input": {"text": utterance, "context": context},
            "application": {"initial_state": {"task_id": tid, "checkpoint_calls": calls}},
            "expected": {
                "answer": "最新の変更範囲と条件を整理し、新しい内容への同意を求める。まだ書き込まない。",
                "delegation": "required",
                "golden_path": {"turns": 2},
            },
        }
        return scenario


def build_checkpoints():
    scenarios = [create_checkpoint(tid) for tid in ("21", "22")]
    save(DATA / "crawl.json", {"schema_version": "1.0", "scenarios": scenarios})
    import copy

    walk = copy.deepcopy(scenarios)
    for s in walk:
        s["input"]["recordings"] = [
            {
                "id": condition,
                "condition": condition,
                # Relative to this dataset file, as the harness resolves recording paths.
                "path": f"../../artifacts/retail/name-phone-v1/audio/{s['id']}_{condition}.wav",
                "metadata": {"source": "synthetic", "seed": 41},
            }
            for condition in ("clean", "noisy")
        ]
    save(DATA / "walk.json", {"schema_version": "1.0", "scenarios": walk})
