"""Versioned synthetic identity overlay; upstream DB stays unchanged."""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_VERSION = "name-phone-v1"
LEGACY_LOOKUPS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
NAME_MATCH_VERSION = "registered-kana-v1"
# name-kana variant: names match registered readings only, never the kanji spelling.
READING_ONLY_MATCH = "reading-v1"
# Explicit synthetic readings for the three pilot customers; do not infer readings
# from kanji or user IDs for customers without registered readings.
NAME_READINGS = {
    "akemi_sasaki_1261": {"first_name": "あけみ", "last_name": "ささき"},
    "kyosuke_hasegawa_4516": {"first_name": "きょうすけ", "last_name": "はせがわ"},
    "kana_tanaka_8020": {"first_name": "かな", "last_name": "たなか"},
}


def normalize_name(value):
    value = re.sub(r"\s", "", unicodedata.normalize("NFKC", value))
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in value)


def identity_matches(uid, row, first_name, last_name, phone_number, reading_only=False):
    if normalize(row["phone_number"]) != normalize(phone_number):
        return False
    readings = NAME_READINGS.get(uid, {})
    return all(
        bool(normalize_name(value))
        and normalize_name(value)
        in {normalize_name(readings.get(key, ""))} | (set() if reading_only else {normalize_name(row[key])})
        for key, value in (("first_name", first_name), ("last_name", last_name))
    )


def normalize(value):
    return re.sub(r"[\s\-()ー−]", "", unicodedata.normalize("NFKC", value))


@lru_cache(maxsize=1)
def identities(profile="baseline"):
    users = json.loads((ROOT / "vendor/j_tau/data/tau2/domains/retail_ja/db.json").read_text())["users"]
    records = {
        uid: {
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "phone_number": f"050-0000-{i:04d}",
        }
        for i, (uid, user) in enumerate(sorted(users.items()), 1)
    }

    if profile.startswith("sequence-") and profile.endswith("-v1"):
        from voice_eval.retail.phone_sequence import NEW_PHONE, TASK_USERS

        tid = profile.removeprefix("sequence-").removesuffix("-v1")
        if tid not in TASK_USERS:
            raise ValueError("Unknown phone fixture profile")
        records[TASK_USERS[tid]]["phone_number"] = NEW_PHONE
    elif profile != "baseline":
        raise ValueError("Unknown phone fixture profile")
    return records


def adapt_task(t):
    users = json.loads((ROOT / "vendor/j_tau/data/tau2/domains/retail_ja/db.json").read_text())["users"]
    uid = None
    for action in t["evaluation_criteria"].get("actions", []):
        args = action["arguments"]
        if args.get("user_id") in users:
            uid = args["user_id"]
            break
        if action["name"] in LEGACY_LOOKUPS:
            matches = [
                key
                for key, u in users.items()
                if (
                    args.get("email") == u["email"]
                    if "email" in args
                    else args.get("first_name") == u["name"]["first_name"]
                    and args.get("last_name") == u["name"]["last_name"]
                    and args.get("zip") == u["address"]["zip"]
                )
            ]
            if len(matches) == 1:
                uid = matches[0]
                break
        if args.get("order_id"):
            db = json.loads((ROOT / "vendor/j_tau/data/tau2/domains/retail_ja/db.json").read_text())
            uid = db["orders"].get(args["order_id"], {}).get("user_id")
            if uid:
                break
    if not uid:
        known = normalize(str(t["user_scenario"]["instructions"].get("known_info", "")))
        matches = [
            key
            for key, u in users.items()
            if key in known
            or normalize(u["email"]) in known
            or normalize(u["name"]["last_name"] + u["name"]["first_name"]) in known
        ]
        if len(matches) > 1:
            matches = [key for key in matches if normalize(users[key]["address"]["zip"]) in known]
        if len(matches) == 1:
            uid = matches[0]
    if not uid:
        raise ValueError(f"Cannot resolve identity for task {t['id']}")
    identity = identities()[uid]
    instructions = t["user_scenario"]["instructions"]
    if not isinstance(instructions, dict):
        raise TypeError("Expected structured caller instructions")
    instructions["known_info"] = (instructions.get("known_info") or "") + (
        f"\n今回の本人確認は氏名＋登録電話番号です。氏名は{identity['last_name']}{identity['first_name']}、"
        f"電話番号は{identity['phone_number']}です。求められたら正確に答えてください。"
        "メールアドレス・郵便番号を本人確認に使いません。電話番号は実験用の架空データです。"
    )
    for action in t["evaluation_criteria"].get("actions", []):
        if action["name"] in LEGACY_LOOKUPS:
            action["name"] = "find_user_id_by_name_phone"
            action["arguments"] = dict(identity)
            if action.get("compare_args"):
                action["compare_args"] = list(identity)
    return t
