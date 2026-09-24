"""Source tasks and Japanese pilot; evaluator expectations never enter agent prompts."""

import copy
import hashlib
import json
import uuid
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor/j_tau/data/tau2/domains/retail_ja"
DATA = ROOT / "data/retail"
PILOT = ("control", "21", "22", "10", "11", "5")
COMMIT = "36c77f84438031b9454047102bbbcab71d3ad262"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def tasks():
    result = {str(t["id"]): t for t in json.loads((SOURCE / "tasks.json").read_text())}
    control = copy.deepcopy(result["22"])
    control["id"] = "control"
    control["initial_state"] = None
    control["user_scenario"] = {
        "persona": "簡潔で協力的な日本語の顧客。",
        "instructions": {
            "domain": "retail_ja",
            "reason_for_call": "保留中の注文#W9911714の配送先を変更したい。",
            "known_info": "佐々木明美、登録郵便番号689-0831。新住所は〒100-0001、日本、東京都、千代田区、丸の内1-2-3、建物名なし。",
            "unknown_info": None,
            "task_instructions": "本人確認に回答し、配送先変更の内容が正しければ明示的に承認する。登録住所の変更は求めない。完了案内を確認する。",
        },
    }
    control["evaluation_criteria"]["actions"] = [
        a for a in result["22"]["evaluation_criteria"]["actions"] if a["name"] == "modify_pending_order_address"
    ]
    control["evaluation_criteria"]["nl_assertions"] = []
    control["evaluation_criteria"]["communicate_info"] = []
    control["evaluation_criteria"]["reward_basis"] = ["DB"]
    result["control"] = control
    from voice_eval.retail.identity import adapt_task

    return {key: adapt_task(value) for key, value in result.items()}


def task(task_id):
    key = str(task_id).removeprefix("retail-")
    try:
        return copy.deepcopy(tasks()[key])
    except KeyError:
        raise ValueError(f"Unknown retail task: {task_id}") from None


def private_caller(t):
    return json.dumps(t["user_scenario"], ensure_ascii=False)


def initial(task_id):
    return {"task_id": str(task_id)}


def build():
    from voice_eval.retail.executor import RetailExecutor

    DATA.mkdir(parents=True, exist_ok=True)
    from voice_eval.retail.identity import IDENTITY_VERSION, NAME_MATCH_VERSION, NAME_READINGS, identities

    save(
        DATA / "identity.json",
        {
            "version": IDENTITY_VERSION,
            "synthetic": True,
            "users": identities(),
            "name_match_version": NAME_MATCH_VERSION,
            "registered_name_readings": NAME_READINGS,
        },
    )
    with RetailExecutor(task_id="21") as ex:
        save(DATA / "tools.json", [{"type": "function", **s["function"], "strict": False} for s in ex.schemas])
    save(DATA / "facts.json", {"domain": "retail_ja", "language": "ja-JP"})
    policy = (SOURCE / "policy.md").read_text().replace("メールアドレス、または氏名＋郵便番号", "氏名＋登録電話番号")
    supplement = """
日本語でのみ対応する。すべての業務に同じ15公開ツールを使用する。
毎回、氏名と登録電話番号で本人確認する。メールアドレスや郵便番号は本人確認に使わない。自称IDだけでは省略しない。
顧客の最新の意向を優先し、操作対象・内容・金額・支払先を説明して明示的な同意を得る。
内容が変われば同意を取り直す。注文の一度限りの変更は全対象をまとめる。
登録住所と各注文の配送先を区別し、依頼のない変更や巻き戻しはしない。
返品・交換の申請と物理的な返送・返金完了を区別する。
明示的に有人対応を希望された場合もtransfer_to_human_agentsを実行する。
ツールは1つずつ実行する。音声フロントエンドは聞き取りと短い待機案内を続けてよい。
結果が返る前に完了を案内しない。結果のok=falseは失敗として扱う。
確認中の操作(pending_action)を会話の中で保持し、訂正があれば内容を更新して再確認する。
評価シナリオのIDから顧客の要求を推測しない。ツール名や内部情報は顧客に説明しない。
"""
    prompts = ROOT / "prompts/retail/baseline"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "backend.txt").write_text(policy + "\n\n## この音声実験の運用追補\n" + supplement)
    (prompts / "frontend.txt").write_text(
        "あなたは日本語の小売カスタマーサポート窓口です。業務判断・照会・変更は全てbackendへ委譲してください。\n顧客の発言、訂正、同意、拒否を省略せず伝えます。本人確認や操作結果を推測しません。\nbackendが求める確認を顧客に伝え、最新の内容への同意を得ます。短く自然な日本語で話し、ツール名は読み上げません。\n"
    )
    scenarios = []
    for tid in tasks():
        t = task(tid)
        instructions = t["user_scenario"]["instructions"]
        reason = instructions["reason_for_call"] if isinstance(instructions, dict) else instructions
        known = instructions.get("known_info", "") if isinstance(instructions, dict) else ""
        scenarios.append(
            {
                "id": f"retail-{tid}",
                "title": f"日本語retail {tid}",
                "type": "retail",
                "interaction": "multi_turn",
                "tags": ["ja", "retail", "pilot"],
                "input": {"text": reason},
                "application": {"initial_state": initial(tid)},
                "expected": {
                    "answer": "最新の依頼を業務ポリシーに従って処理し、実際の結果を日本語で案内する。",
                    "delegation": "required",
                    "golden_path": {
                        "delegations": 4,
                        "steps": [
                            "依頼を伝える",
                            "本人確認方法を案内する",
                            "本人確認に答える",
                            "注文の状況を照会する",
                            "対象と可能な操作を説明する",
                            "最新の希望を伝える",
                            "確定内容を確認する",
                            "同意を伝える",
                            "処理結果を案内する",
                        ],
                    },
                },
                "simulation_parameters": {
                    "goal": private_caller(t),
                    "known_facts": {"本人情報": known or reason},
                    "persona": {
                        "description": "元タスクの条件付き行動を守る日本語の顧客。",
                        "speech_instructions": "日本語で自然に話してください。",
                        "backchannels": ["はい", "なるほど"],
                    },
                    "agenda": [
                        {
                            "id": "identity",
                            "commitment": "求められた本人情報を伝える",
                            "trigger_condition": "本人確認を求められたとき",
                            "completion_condition": "必要な情報を伝えた",
                            "action": "answer",
                            "facts": ["本人情報"],
                        },
                        {
                            "id": "request",
                            "commitment": private_caller(t),
                            "trigger_condition": "会話の状況に応じて元タスクの条件を満たしたとき",
                            "completion_condition": "元タスクの条件付き変更を全て伝えて最終結果を確認した",
                            "completion_basis": "live_context",
                            "action": "correct",
                        },
                    ],
                    "expectations": ["日本語のみ。元タスクにない情報を創作しない。条件付き変更を省略しない。"],
                },
            }
        )
    save(DATA / "all_run.json", {"schema_version": "1.0", "scenarios": scenarios})
    save(
        DATA / "run.json",
        {
            "schema_version": "1.0",
            "scenarios": [s for s in scenarios if s["application"]["initial_state"]["task_id"] in PILOT],
        },
    )
    manifest = {
        "repository": "sbintuitions/j-tau-bench",
        "commit": COMMIT,
        "language": "ja-JP",
        "license": "vendor/j_tau/LICENSE",
        "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()},
        "adaptations": [
            "Isolated worker namespace bootstrap skips eager runner imports",
            "Japanese voice operational supplement",
            "name-phone-v1: synthetic phone sidecar; two legacy lookups replaced by one name+phone lookup",
            "Scores are not official J-tau scores",
        ],
    }
    save(DATA / "source.json", manifest)


if __name__ == "__main__":
    build()
