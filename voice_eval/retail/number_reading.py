"""Lossless identifier readings for an opt-in voice experiment, not monetary values."""

import copy
import re
import unicodedata

DIGITS = ("ゼロ", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ")
RULE = """電話番号・郵便番号・商品ID・注文番号は数量ではなく識別子です。
発話用のカタカナ読みがある場合、その順序と各桁を省略せず読みます。
「・」では桁を区切り、「／」では短い間を置き、区切り記号自体は読みません。
0=ゼロ、1=イチ、2=ニ、3=サン、4=ヨン、5=ゴ、6=ロク、7=ナナ、8=ハチ、9=キュウ。
復唱への同意は全桁が一致する場合だけ行い、違う場合は具体的な位置と正しい読みを伝えます。
金額・個数・日時にはこの読み方を適用しません。
"""
# Deliberately bounded to the existing fixture formats. No blanket number replacement.
IDENTIFIER = re.compile(r"(?<![A-Za-z0-9])(?:#?W\d{7}|0\d{2}-\d{4}-\d{4}|\d{3}-\d{4}|\d{10})(?![\d円年月日時分個])")


def format_identifier(value):
    text = unicodedata.normalize("NFKC", value).strip()
    if not re.fullmatch(r"(?:#?W)?[0-9]+(?:-[0-9]+)*", text):
        raise ValueError("数字列またはWで始まる注文番号を指定してください。推測変換はしません。")
    prefix = "シャープ・ダブリュー・" if text.startswith("#W") else "ダブリュー・" if text.startswith("W") else ""
    body = text.removeprefix("#W").removeprefix("W")
    spoken = prefix + "／".join("・".join(DIGITS[int(c)] for c in part) for part in body.split("-"))
    return {
        "canonical": text,
        "digits": body.replace("-", ""),
        "spoken": spoken,
        "digit_count": len(body.replace("-", "")),
    }


def caller_projection(scenario):
    result = copy.deepcopy(scenario)
    mappings = {}

    def project(value):
        if isinstance(value, str):

            def replace(match):
                original = match.group()
                mappings[original] = format_identifier(original)
                return "「" + mappings[original]["spoken"] + "」"

            return IDENTIFIER.sub(replace, value)
        if isinstance(value, dict):
            return {key: project(item) for key, item in value.items()}
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    # Private caller fields only. Expected outcomes and backend tools never enter the prompt.
    result["simulation_parameters"] = project(result["simulation_parameters"])
    result["input"]["text"] = project(result["input"]["text"])
    persona = result["simulation_parameters"]["persona"]
    persona["speech_instructions"] += "\n" + RULE
    return result, list(mappings.values())
