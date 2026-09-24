"""Build paired Japanese/English scenarios without disclosing grading answers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-06"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def arguments(language: str, time: str = "19:00") -> dict:
    return {"guest_name": "マヤ" if language == "ja" else "Maya", "date": DATE, "time": time, "party_size": 2}


def call(name: str, args: dict) -> dict:
    return {"name": name, "arguments": args}


def expected_booking(args: dict) -> dict:
    slot = {key: value for key, value in args.items() if key != "guest_name"}
    return {
        "answer": "Check availability, create exactly one reservation for the requested details, then confirm success.",
        "criteria": [
            "Preserve the caller's latest intent and all supplied details.",
            "Check availability before creating the reservation; never book an unavailable slot.",
            "Create exactly one reservation; never claim success before the tool succeeds.",
            "Respond in the caller's language and accurately communicate the completed reservation.",
        ],
        "delegation": "required",
        "tools": {
            "required": [call("check_availability", slot), call("create_reservation", args)],
            "prohibited": [call("cancel_reservation", {})],
        },
        "state": {
            "reservation_created": True,
            **args,
            "reservations": [{"reservation_id": "R-001", **args, "cancelled": False}],
        },
        "golden_path": {"turns": 2, "delegations": 1},
    }


def single(language: str, case: str) -> dict:
    texts = {
        "ja": {
            "booking": "2026年10月6日の午後7時に、2名で予約してください。名前はマヤです。",
            "correction": "2026年10月7日、すみません、10月6日の午後7時に、2名で予約してください。名前はマヤです。",
            "missing_name": "2026年10月6日の午後7時に、2名で予約してください。",
            "availability": "2026年10月6日の午後7時、2名の空席はありますか？空席の確認だけで、まだ予約はしないでください。",
        },
        "en": {
            "booking": "Please book a table for two under Maya on October 6, 2026, at 7 p.m.",
            "correction": "Please book a table for two under Maya on October 7, sorry, October 6, 2026, at 7 p.m.",
            "missing_name": "Please book a table for two on October 6, 2026, at 7 p.m.",
            "availability": "Do you have a table for two on October 6, 2026, at 7 p.m.? Just check availability; don't book yet.",
        },
    }
    args = arguments(language)
    expected = expected_booking(args)
    if case in {"availability", "missing_name"}:
        expected["state"] = {"unchanged": True}
        expected["tools"] = {
            "required": [],
            "prohibited": [call("create_reservation", {}), call("cancel_reservation", {})],
        }
        if case == "availability":
            expected["tools"]["required"] = [
                call("check_availability", {k: v for k, v in args.items() if k != "guest_name"})
            ]
            expected["answer"] = "Report availability without making a reservation."
            expected["criteria"] = [
                "Check the requested slot and report the result in the caller's language.",
                "Do not book.",
            ]
        else:
            expected["delegation"] = "forbidden"
            expected["golden_path"]["delegations"] = 0
            expected["answer"] = "Ask for the missing reservation name, without guessing or making a reservation."
            expected["criteria"] = [
                "Ask for the guest name in the caller's language.",
                "Do not call tools or invent a name.",
            ]
    return {
        "id": f"{case}_{language}",
        "title": f"{case} ({language})",
        "type": "clarification" if case == "missing_name" else "availability" if case == "availability" else "booking",
        "interaction": "single_turn",
        "tags": [language, case],
        "input": {"text": texts[language][case]},
        "application": {"initial_state": {"reservations": []}},
        "expected": expected,
    }


def multi(language: str, alternative: bool) -> dict:
    ja = language == "ja"
    args = arguments(language, "20:00" if alternative else "19:00")
    expected = expected_booking(args)
    expected["golden_path"] = {"turns": 7, "delegations": 2 if alternative else 1}
    expected["criteria"] += ["Collect missing information without guessing."]
    slot = {k: v for k, v in args.items() if k != "guest_name"}
    steps = []
    if alternative:
        unavailable = {**slot, "time": "18:00"}
        expected["tools"]["required"].insert(0, call("check_availability", unavailable))
        expected["criteria"] += [
            "The caller must accept 20:00 before the reservation is created; do not infer consent from availability."
        ]
        steps.append(
            {
                "id": "first_check",
                "kind": "tool",
                "tool": "check_availability",
                "arguments": unavailable,
                "critical": True,
            }
        )
    steps += [
        {
            "id": "availability",
            "kind": "tool",
            "tool": "check_availability",
            "arguments": slot,
            "critical": True,
            "after": ["first_check"] if alternative else [],
        },
        {
            "id": "book",
            "kind": "tool",
            "tool": "create_reservation",
            "arguments": args,
            "after": ["availability"],
            "critical": True,
        },
        {"id": "state", "kind": "state", "arguments": expected["state"], "after": ["book"], "critical": True},
        {"id": "confirmation", "kind": "grounded_confirmation", "after": ["book"], "critical": True},
    ]
    expected["procedure"] = {"id": "check_then_book", "steps": steps}
    details = "マヤ、2名です。" if ja else "Under Maya, for two."
    agenda = [
        {
            "id": "details",
            "commitment": "Supply your name and party size when asked.",
            "trigger_condition": "Assistant requests name or party size.",
            "completion_condition": "Name and party size have been spoken.",
            "action": "answer",
            "facts": ["guest_name", "party_size"],
            "response_hint": details,
        }
    ]
    if alternative:
        agenda += [
            {
                "id": "alternative",
                "commitment": "After hearing that 18:00 is unavailable, explicitly ask to book 20:00 instead.",
                "trigger_condition": "Assistant says the initial 18:00 slot is unavailable.",
                "completion_condition": "Caller explicitly requests a 20:00 booking.",
                "action": "answer",
                "facts": ["time"],
                "response_hint": "では午後8時で予約してください。" if ja else "Then please book 8 p.m. instead.",
            }
        ]
        opening = (
            "2026年10月6日の午後6時に予約したいです。"
            if ja
            else "I'd like to book a table on October 6, 2026, at 6 p.m."
        )
    else:
        agenda += [
            {
                "id": "time",
                "commitment": "Provide the desired 19:00 time when asked.",
                "trigger_condition": "Assistant asks for the time.",
                "completion_condition": "Caller supplies 19:00.",
                "action": "answer",
                "facts": ["time"],
                "response_hint": "午後7時です。" if ja else "At 7 p.m., please.",
            }
        ]
        opening = "2026年10月6日に予約したいです。" if ja else "I'd like to book a table on October 6, 2026."
    agenda.append(
        {
            "id": "finish",
            "commitment": "Thank the assistant once the correct reservation is confirmed.",
            "trigger_condition": "The requested booking is confirmed.",
            "completion_condition": "Caller understands the booking is complete.",
            "completion_basis": "live_context",
            "action": "finish",
            "required": False,
        }
    )
    return {
        "id": f"{'alternative' if alternative else 'collect'}_{language}",
        "title": f"{'Alternative slot' if alternative else 'Collect details'} ({language})",
        "type": "booking",
        "interaction": "multi_turn",
        "tags": [language, "alternative" if alternative else "collect"],
        "input": {"text": opening},
        "application": {"initial_state": {"reservations": []}},
        "expected": expected,
        "simulation_parameters": {
            "goal": f"Book exactly one table for {args}. Speak only {'Japanese' if ja else 'English'}.",
            "known_facts": args,
            "agenda": agenda,
            "expectations": [
                "Answer the assistant's actual questions naturally; give multiple requested details together.",
                "Do not reveal your alternative time until the initial slot is reported unavailable."
                if alternative
                else "Provide missing details when asked.",
            ],
            "persona": {
                "id": f"diner_{language}",
                "description": f"A concise {'Japanese' if ja else 'English'} speaking diner.",
                "voice": "cedar",
                "speech_instructions": f"Speak only {'Japanese' if ja else 'English'}, naturally and clearly.",
                "backchannels": ["はい"] if ja else ["Right."],
                "backchannel_tendency": 0.12,
            },
        },
    }


def build() -> None:
    crawl = [
        single(lang, case)
        for lang in ("ja", "en")
        for case in ("booking", "correction", "missing_name", "availability")
    ]
    walk = []
    for lang in ("ja", "en"):
        for condition in ("clean", "noisy"):
            scenario = copy.deepcopy(single(lang, "correction"))
            scenario["id"] = f"correction_{lang}_{condition}"
            scenario["input"]["recordings"] = [
                {
                    "id": condition,
                    "path": f"../artifacts/audio/correction_{lang}_{condition}.wav",
                    "condition": condition,
                    "metadata": {
                        "source": "synthetic",
                        "language": lang,
                        "noise_seed": 41,
                        "description": "Gemini TTS fixture; not a human recording",
                    },
                }
            ]
            walk.append(scenario)
    for mode, scenarios in (
        ("crawl", crawl),
        ("walk", walk),
        ("run", [multi(lang, alt) for lang in ("ja", "en") for alt in (False, True)]),
    ):
        save(ROOT / "data" / f"{mode}.json", {"schema_version": "1.0", "scenarios": scenarios})
    save(
        ROOT / "data" / "restaurant_facts.json",
        {
            "business_name": "こもれび食堂 / Komorebi Dining",
            "reference_year": 2026,
            "timezone": "Asia/Tokyo",
            "hours": "17:00–22:00 daily",
            "parking": "No parking",
            "availability_overrides": [{"date": DATE, "time": "18:00", "party_size": 2, "available": False}],
        },
    )


if __name__ == "__main__":
    build()
