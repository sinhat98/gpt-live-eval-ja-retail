"""Isolated J-tau JSONL worker. stdout is reserved for RPC responses."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor/j_tau/src/tau2"
# Import the pinned business modules without tau2's eager runner/voice imports.
# Upstream files are preserved verbatim; the runner is not used by this adapter.
for name in ("tau2", "tau2.domains", "tau2.orchestrator", "tau2.evaluator"):
    module = types.ModuleType(name)
    module.__path__ = [str(SOURCE.joinpath(*name.split(".")[1:]))]
    sys.modules[name] = module

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.domains.retail_ja.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

sys.path.insert(0, str(ROOT))
from tau2.domains.retail_ja.tools import RetailTools
from tau2.environment.toolkit import ToolType, is_tool

from voice_eval.retail.identity import LEGACY_LOOKUPS, READING_ONLY_MATCH, identities, identity_matches

upstream_environment = get_environment
READING_ONLY_LOOKUP = {
    "description": "氏名の読みがなと登録電話番号の両方を照合する。\n\n"
    "音声で聞いた氏名は漢字に変換せず、聞いた読みをひらがなで指定する。漢字表記では照合しない。部分一致では本人確認しない。",
    "first_name": "顧客の名の読みがな（ひらがな）。漢字に変換しない。",
    "last_name": "顧客の姓の読みがな（ひらがな）。漢字に変換しない。",
}


def lookup_schema(schema):
    """Describe the reading-only name match to the backend, only when it is enabled."""
    if (
        os.getenv("RETAIL_NAME_MATCH") != READING_ONLY_MATCH
        or schema["function"]["name"] != "find_user_id_by_name_phone"
    ):
        return schema
    function = schema["function"]
    function["description"] = READING_ONLY_LOOKUP["description"]
    for key in ("first_name", "last_name"):
        function["parameters"]["properties"][key]["description"] = READING_ONLY_LOOKUP[key]
    return schema


class PhoneRetailTools(RetailTools):
    @is_tool(ToolType.READ)
    def format_spoken_identifier(self, value: str) -> dict:
        """会話で受け取った識別子の数字列を、省略なしのカタカナ読みに変換する。顧客の正解情報は照会しない。

        Args:
            value: 電話番号・郵便番号・商品ID・Wで始まる注文番号の数字表記。金額や日時は渡さない。
        """
        from voice_eval.retail.number_reading import format_identifier

        return format_identifier(value)

    @is_tool(ToolType.READ)
    def find_user_id_by_name_phone(self, first_name: str, last_name: str, phone_number: str) -> str:
        """氏名（登録表記または登録済みの読みかな）と登録電話番号の両方を照合する。
        音声で聞いた氏名は漢字を推測せず、読みかなで指定できる。部分一致では本人確認しない。

        Args:
            first_name: 顧客の名。登録表記または読みかな（ひらがな・カタカナ）。
            last_name: 顧客の姓。登録表記または読みかな（ひらがな・カタカナ）。
            phone_number: 登録電話番号。数字・ハイフン・空白・括弧で指定。
        """
        reading_only = os.getenv("RETAIL_NAME_MATCH") == READING_ONLY_MATCH
        matches = [
            uid
            for uid, row in identities(os.getenv("RETAIL_PHONE_PROFILE", "baseline")).items()
            if identity_matches(uid, row, first_name, last_name, phone_number, reading_only)
        ]
        if len(matches) != 1:
            raise ValueError("氏名と電話番号に一致する顧客が一意に見つかりません。再確認してください。")
        return matches[0]


def get_environment(**kwargs):
    env = upstream_environment(**kwargs)
    env.tools = PhoneRetailTools(env.tools.db)
    env.policy = env.policy.replace("メールアドレス、または氏名＋郵便番号", "氏名＋登録電話番号")
    return env


def plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


class Worker:
    def __init__(self):
        self.env = None
        self.task = None
        self.calls = {}
        self.history = []

    def fresh(self):
        env = get_environment()
        initial = self.task.initial_state
        env.set_state(
            initialization_data=initial.initialization_data if initial else None,
            initialization_actions=initial.initialization_actions if initial else None,
            message_history=initial.message_history or [] if initial else [],
        )
        return env

    def dispatch(self, method, params):
        if method == "initialize":
            tasks = {str(t.id): t for t in get_tasks(task_split_name=None)}
            custom = params.get("task")
            self.task = Task.model_validate(custom) if custom else tasks[str(params["task_id"])]
            self.env = self.fresh()
            self.calls = {}
            self.history = list(self.task.initial_state.message_history or []) if self.task.initial_state else []
            return {"task_id": self.task.id, "policy": self.env.policy}
        if self.env is None:
            raise ValueError("initialize must be called first")
        if method == "tool_schema":
            return [
                lookup_schema(t.openai_schema)
                for t in self.env.get_tools()
                if t.name not in LEGACY_LOOKUPS
                and (t.name != "format_spoken_identifier" or os.getenv("RETAIL_NUMBER_READING") == "katakana-v1")
            ]
        if method == "snapshot":
            return self.env.tools.db.model_dump(mode="json")
        if method == "execute":
            call_id, name, arguments = params["call_id"], params["name"], params["arguments"]
            if call_id in self.calls:
                previous = self.calls[call_id]
                if previous["name"] != name or previous["arguments"] != arguments:
                    raise ValueError("call_id reused with different arguments")
                return previous["output"]
            try:
                if name == "format_spoken_identifier" and os.getenv("RETAIL_NUMBER_READING") != "katakana-v1":
                    raise ValueError("Identifier reading experiment is not enabled")
                if name in LEGACY_LOOKUPS:
                    raise ValueError("本人確認は氏名と電話番号を使用してください。")
                result = plain(self.env.make_tool_call(name, **arguments))
                output = {"ok": True, "result": result}
            except (ValueError, TypeError) as exc:
                output = {"ok": False, "error": str(exc)}
            self.calls[call_id] = {"name": name, "arguments": arguments, "output": output}
            self.history.extend(
                [
                    AssistantMessage(
                        role="assistant", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)]
                    ),
                    ToolMessage(
                        role="tool",
                        id=call_id,
                        content=json.dumps(output["result"], ensure_ascii=False)
                        if output["ok"]
                        else "Error: " + output["error"],
                        error=not output["ok"],
                    ),
                ]
            )
            return output
        if method == "evaluate":
            gold = self.fresh()
            # Unlike upstream warning-only gold replay, fail preflight on any error.
            for action in self.task.evaluation_criteria.actions or []:
                gold.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
            result = EnvironmentEvaluator.calculate_reward(get_environment, self.task, self.history, env_kwargs={})
            replay = get_environment()
            initial = self.task.initial_state
            replay.set_state(
                initial.initialization_data if initial else None,
                initial.initialization_actions if initial else None,
                self.history,
            )
            parity = replay.tools.db.model_dump(mode="json") == self.env.tools.db.model_dump(mode="json")
            return {
                "environment": plain(result),
                "replay_matches_snapshot": parity,
                "reward_basis": self.task.evaluation_criteria.reward_basis,
                "nl_assertions": self.task.evaluation_criteria.nl_assertions,
                "task_reward": None,
                "semantic_evaluation": "pending",
            }
        if method == "close":
            return {"closed": True}
        raise ValueError("Unknown RPC method")


def main():
    worker = Worker()
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            result = worker.dispatch(request["method"], request.get("params", {}))
            response = {"request_id": request["request_id"], "result": result}
        except (ValueError, KeyError) as exc:
            response = {
                "request_id": request.get("request_id"),
                "error": {"kind": "business_error", "message": str(exc)},
            }
        except Exception as exc:  # noqa: BLE001 - classify and persist trial/worker failures
            response = {
                "request_id": request.get("request_id"),
                "error": {"kind": "infra_error", "message": f"{type(exc).__name__}: {exc}"},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)
        if request.get("method") == "close":
            break


if __name__ == "__main__":
    main()
