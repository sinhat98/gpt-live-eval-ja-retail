"""Text comparison with the same policy, backend model, and full tool set."""

import json
import os
import time

from voice_eval.retail.data import ROOT, private_caller, save, task
from voice_eval.retail.executor import RetailExecutor
from voice_eval.retail.scoring import score

CALLER = """あなたは日本語の顧客シミュレーターです。オペレーターではありません。
私的な元タスクに従い、客の発言だけを自然な日本語で生成してください。
既知情報以外を創作せず、本人確認には求められた情報だけ答えてください。
条件付きの追加・撤回・意向変更は指定タイミングで必ず実行してください。
発言の最初で将来の意向変更を漏らさず、変更した操作に自動で同意しないでください。
全ての条件付き行動を済ませ、最終結果を聞いてから、終了時のみ末尾に<END>を付けてください。
"""


def run_text(tid, output, *, variant="baseline", duration=180):
    from openai import OpenAI

    t = task(tid)
    policy = (ROOT / f"prompts/retail/{variant}/backend.txt").read_text()
    client = OpenAI(max_retries=0, timeout=45)
    deadline = time.monotonic() + duration
    transcript, usage, backend_input = [], [], []
    status = "task_timeout"
    with RetailExecutor(task_id=tid, task=t if tid == "control" else None, log_dir=output) as executor:
        save(output / "initial_state.json", executor.initial_snapshot)
        save(output / "tools.json", executor.schemas)
        (output / "backend.txt").write_text(policy)
        save(output / "caller.json", t["user_scenario"])
        tools = [{"type": "function", **s["function"], "strict": False} for s in executor.schemas]
        try:
            for turn in range(60):
                if time.monotonic() >= deadline:
                    break
                caller = client.with_options(timeout=max(1, min(45, deadline - time.monotonic()))).responses.create(
                    model=os.getenv("OPENAI_CALLER_BACKEND_MODEL", "gpt-5.6-luna"),
                    reasoning={"effort": "low"},
                    instructions=CALLER + "\n私的タスク:\n" + private_caller(t),
                    input=json.dumps(
                        [m for m in transcript if m["role"] in ("user", "assistant")]
                        or [{"role": "assistant", "content": "お問い合わせをどうぞ。"}],
                        ensure_ascii=False,
                    ),
                )
                usage.append({"component": "caller", "usage": caller.usage.model_dump()})
                text = caller.output_text
                end = "<END>" in text
                text = text.replace("<END>", "").strip()
                if text:
                    transcript.append({"role": "user", "content": text})
                    backend_input.append({"role": "user", "content": text})
                for _ in range(30):
                    if time.monotonic() >= deadline:
                        break
                    response = client.with_options(
                        timeout=max(1, min(45, deadline - time.monotonic()))
                    ).responses.create(
                        model=os.getenv("OPENAI_LIVE_BACKEND_MODEL", "gpt-5.6-terra"),
                        instructions=policy,
                        input=backend_input,
                        tools=tools,
                        parallel_tool_calls=False,
                        reasoning={"effort": os.getenv("OPENAI_LIVE_BACKEND_REASONING_EFFORT", "none")},
                    )
                    usage.append({"component": "backend", "usage": response.usage.model_dump()})
                    backend_input.extend(response.output)
                    calls = [item for item in response.output if item.type == "function_call"]
                    for call in calls:
                        arguments = json.loads(call.arguments)
                        result = executor.execute(call.name, arguments, call_id=call.call_id)
                        backend_input.append(
                            {
                                "type": "function_call_output",
                                "call_id": call.call_id,
                                "output": json.dumps(result, ensure_ascii=False),
                            }
                        )
                        transcript.append({"role": "tool", "name": call.name, "arguments": arguments, "output": result})
                    if response.output_text:
                        transcript.append({"role": "assistant", "content": response.output_text})
                    if not calls:
                        break
                save(output / "transcript.json", transcript)
                if end:
                    status = "completed"
                    break
            save(output / "transcript.json", transcript)
            if status == "completed":
                try:
                    result = score(executor, t, transcript, client)
                except Exception as exc:  # noqa: BLE001 - classify and persist trial/worker failures
                    result = {"status": "grader_error", "error_type": type(exc).__name__, "passed": False}
            else:
                result = {"status": status, "passed": False, "task_reward": 0.0}
        except Exception as exc:  # noqa: BLE001 - classify and persist trial/worker failures
            result = {
                "status": "task_timeout" if time.monotonic() >= deadline else "infra_error",
                "error_type": type(exc).__name__,
                "passed": False,
            }
        finally:
            from voice_eval.retail.artifacts import state_diff

            save(output / "transcript.json", transcript)
            save(output / "executions.json", executor.executions)
            save(output / "final_state.json", executor.snapshot())
            save(output / "state_diff.json", state_diff(executor.initial_snapshot, executor.snapshot()))
            save(output / "usage.json", usage)
        result.update(
            {"mode": "text", "task_id": tid, "variant": variant, "confirmed_completion_ms": None, "cost_usd": None}
        )
        result["caller_reasoning_effort"] = "low"
        result["duration_seconds"] = time.monotonic() - (deadline - duration)
        save(output / "result.json", result)
        return result
