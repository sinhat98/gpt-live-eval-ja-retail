"""Resume the predeclared RUN comparison, at most three attempts per variant."""

import argparse
import json
import subprocess
import sys

from voice_eval.retail.cli import ARTIFACTS
from voice_eval.retail.data import ROOT, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="21")
    args = parser.parse_args()
    for variant in ("baseline", "candidate"):
        parent = ARTIFACTS / "live/run" / variant / args.scenario
        for _ in range(3):
            completed = list(parent.glob("attempt-*/result.json"))
            if len(completed) >= 3:
                break
            if len(list(parent.glob("attempt-*"))) != len(completed):
                raise SystemExit("Unfinished attempt exists; inspect before resuming")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "voice_eval",
                    "run",
                    "run",
                    "--domain",
                    "retail",
                    "--scenario",
                    args.scenario,
                    "--variant",
                    variant,
                    "--new-attempt",
                ],
                cwd=ROOT,
                check=True,
            )
    rows = []
    for variant in ("baseline", "candidate"):
        for p in sorted((ARTIFACTS / "live/run" / variant / args.scenario).glob("attempt-*/result.json")):
            r = json.loads(p.read_text())
            calls = json.loads((p.parent / "capture/executions.json").read_text())
            details = list(p.parent.glob("harness/**/conversation.result.json"))
            detail = json.loads(details[0].read_text()) if details else {}
            rows.append(
                {
                    "variant": variant,
                    "status": r["status"],
                    "passed": r["passed"],
                    "path": str(p.relative_to(ROOT)),
                    "task_reward": r.get("task_reward"),
                    "termination_reason": r.get("termination_reason"),
                    "authentication_completed": any(
                        c["name"].startswith("find_user_id_by_") and c["output"].get("ok") for c in calls
                    ),
                    "delegations": detail.get("delegation_count"),
                    "caller_validity": r.get("semantic", {}).get("caller_validity"),
                }
            )
    save(
        ARTIFACTS / f"comparison-{args.scenario}.json",
        {
            "mode": "run",
            "task_id": args.scenario,
            "trials_per_variant": 3,
            "rows": rows,
            "interpretation": "Exploratory small sample; inspect caller validity, environment anomalies, and exact settings before attributing an improvement.",
        },
    )


if __name__ == "__main__":
    main()
