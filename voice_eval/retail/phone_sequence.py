"""Change only the selected pilot customer's synthetic phone fixture, retaining baseline prompts."""

OLD_PHONE = "050-0000-0102"
NEW_PHONE = "080-1234-5678"


TASK_USERS = {
    "control": "akemi_sasaki_1261",
    "21": "akemi_sasaki_1261",
    "22": "akemi_sasaki_1261",
    "10": "kyosuke_hasegawa_4516",
    "11": "kyosuke_hasegawa_4516",
    "5": "kana_tanaka_8020",
}


def project(value, tid="5"):
    from voice_eval.retail.identity import identities

    old_phone = identities()[TASK_USERS[str(tid)]]["phone_number"]
    if isinstance(value, str):
        return value.replace(old_phone, NEW_PHONE)
    if isinstance(value, dict):
        return {key: project(item, tid) for key, item in value.items()}
    if isinstance(value, list):
        return [project(item, tid) for item in value]
    return value
