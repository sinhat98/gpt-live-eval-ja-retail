"""Build a local listening index from saved trials; no model calls are made."""

import html
import json
import os
import re
from urllib.parse import quote

from voice_eval.retail.data import ROOT, save


def href(path, base):
    return quote(os.path.relpath(path, base), safe="/")


def transcript_html(text):
    rows = []
    for line in text.splitlines():
        match = re.match(r"^(USER|ASSISTANT) (\d+)(?:\.\.(\d+))?ms(?: \[OVERLAP\])?: (.*)$", line)
        if not match:
            rows.append('<div class="internal">' + html.escape(line) + "</div>")
            continue
        role, start, _end, words = match.groups()
        seconds = int(start) / 1000
        label = "客" if role == "USER" else "オペレーター"
        stamp = f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
        overlap = "（重なりあり）" if "[OVERLAP]" in line else ""
        rows.append(
            f'<div class="utterance"><button data-seek="{seconds}">{stamp}</button> '
            f"<b>{label}</b>{overlap}：{html.escape(words)}</div>"
        )
    return "\n".join(rows)


def main():
    base = ROOT / "artifacts/retail/recordings"
    base.mkdir(exist_ok=True, parents=True)
    selected = set()
    for manifest in base.parent.glob("**/pilot.json"):
        pilot = json.loads(manifest.read_text())
        selected.update(str(ROOT / row["path"]) for row in pilot["rows"])
    cards, rows, index = [], [], ["# 対話音声・文字起こし", "", "[再生ページ](index.html)", ""]
    for result in sorted(base.parent.glob("**/live/**/result.json")):
        trial = result.parent
        relative = trial.relative_to(base.parent).parts
        live_index = relative.index("live")
        version = "/".join(relative[:live_index]) or "legacy"
        parts = relative[live_index + 1 :]
        mode = parts[0]
        label = version + " / " + " / ".join(parts)
        r = json.loads(result.read_text())
        wav = next(iter(sorted(trial.glob("harness/**/conversation.wav"))), None)
        transcript = next(iter(sorted(trial.glob("harness/**/conversation.transcript.txt"))), None)
        text_log = trial / "transcript.json"
        name = version.replace("/", "__") + "__" + "__".join(parts)
        is_pilot = str(result) in selected
        links = [f'<a href="{href(result, base)}">採点結果</a>']
        plain = ""
        body = ""
        if wav:
            links.append(f'<a href="{href(wav, base)}" download>WAV保存</a>')
            body += f'<audio controls preload="none" src="{href(wav, base)}"></audio>'
        if transcript:
            plain = transcript.read_text()
            links.append(f'<a href="{href(transcript, base)}">元の時刻付きログ</a>')
            body += '<div class="transcript">' + transcript_html(plain) + "</div>"
        elif text_log.exists():
            records = json.loads(text_log.read_text())
            plain = "\n\n".join(
                ("客" if turn["role"] == "user" else "オペレーター") + ": " + str(turn.get("content", ""))
                for turn in records
                if turn["role"] in ("user", "assistant")
            )
            body += "<p>テキスト試行のため音声はありません。</p><pre>" + html.escape(plain) + "</pre>"
            links.append(f'<a href="{href(text_log, base)}">元の会話・ツールログ</a>')
        else:
            body += "<p>会話記録なし（開始前エラーなど）。採点結果を参照してください。</p>"
        if plain:
            output = base / f"{name}.txt"
            output.write_text(plain + "\n")
            links.append(f'<a href="{href(output, base)}" download>文字起こしTXT保存</a>')
        badge = "pilot採用" if is_pilot else "比較・開発試行"
        cards.append(
            f'<article data-pilot="{str(is_pilot).lower()}" data-mode="{mode}" data-version="{version}" data-variant="{parts[1]}"><h2>{html.escape(label)}</h2>'
            f"<p>{badge} · {html.escape(r['status'])}</p><nav>{' · '.join(links)}</nav>{body}</article>"
        )
        rows.append(
            {
                "trial": label,
                "identity_version": version,
                "pilot": is_pilot,
                "status": r["status"],
                "audio": href(wav, base) if wav else None,
                "transcript": f"{name}.txt" if plain else None,
                "result": href(result, base),
            }
        )
        index.append(
            f"- {label}：{r['status']} / {badge}"
            + (f" [音声]({href(wav, base)})" if wav else "")
            + (f" [文字起こし]({name}.txt)" if plain else "")
        )
    page = (
        """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>日本語retail 対話記録</title><style>
body{max-width:1100px;margin:32px auto;padding:0 20px;font-family:system-ui;line-height:1.7;background:#f6f7fb;color:#162334}
article{background:white;border:1px solid #d6deeb;border-radius:12px;padding:20px;margin:20px 0}h2{font-size:18px}
a{color:#174f9b}audio{display:block;width:100%;margin:16px 0}.transcript{max-height:460px;overflow:auto}
.utterance{padding:7px 0;border-bottom:1px solid #eee}.internal{font-size:12px;color:#657181;padding:5px;overflow-wrap:anywhere}
body.hide-internal .internal{display:none}button,select{padding:6px;cursor:pointer}pre{white-space:pre-wrap}nav{font-size:14px}
</style><body class="hide-internal"><h1>日本語retail 対話音声・文字起こし</h1>
<p>保存済みの実測記録です。RUNは対話全体、CRAWL/WALKはcheckpoint後の一応答です。
文字起こしはハーネスが記録したテキストで、人による修正・照合は未実施です。発話の分割や聞き取り誤りが残る場合があります。
時刻ボタンで音声をその位置から再生できます。内部ログは発話ではありません。</p>
<nav><a href="#phone-sequence">080-1234-5678への番号変更</a> · <a href="#phone-run">電話番号版のRUN</a> · <a href="#phone">電話番号版すべて</a> · <a href="#all">全試行</a></nav>
<label>表示 <select id="filter">
<option value="phone-sequence">番号変更（080-1234-5678）</option>
<option value="phone-run">氏名＋電話番号版のRUN</option>
<option value="phone">氏名＋電話番号版すべて</option>
<option value="pilot">pilot採用試行（新旧）</option>
<option value="all">全試行</option>
<option value="run">RUN全試行（新旧）</option>
</select></label><span id="count" role="status"></span>
<label><input type="checkbox" id="internal">内部の委譲・処理ログを表示</label>
"""
        + "\n".join(cards)
        + """<script>
const articles=[...document.querySelectorAll('article')];
const select=document.querySelector('#filter');
function filter(){
 const v=select.value;
 articles.forEach(a=>{
  const phone=a.dataset.version==='name-phone-v1';
  const run=a.dataset.mode==='run';
  a.hidden=!(v==='phone-sequence'?a.dataset.variant==='phone-sequence'&&run:v==='phone-run'?phone&&run:v==='phone'?phone:v==='pilot'?a.dataset.pilot==='true':v==='run'?run:true);
 });
 document.querySelector('#count').textContent=` ${articles.filter(a=>!a.hidden).length}件を表示`;
}
function fromHash(){
 const value=location.hash.slice(1);
 select.value=[...select.options].some(o=>o.value===value)?value:'phone-run';
 filter();
}
select.addEventListener('change',()=>{location.hash=select.value;filter()});
window.addEventListener('hashchange',fromHash);fromHash();
document.querySelector('#internal').addEventListener('change',e=>document.body.classList.toggle('hide-internal',!e.target.checked));
document.addEventListener('click',e=>{const b=e.target.closest('[data-seek]');if(!b)return;const a=b.closest('article').querySelector('audio');if(a){a.currentTime=Number(b.dataset.seek);a.play().catch(()=>{});}});
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()})));
</script></body></html>"""
    )
    (base / "index.html").write_text(page)
    (base / "README.md").write_text("\n".join(index) + "\n")
    save(base / "index.json", {"trials": rows, "transcript_source": "saved_harness_text_not_human_verified"})
    print(
        f"{len(rows)} trials; {sum(bool(r['audio']) for r in rows)} audio recordings; {sum(bool(r['transcript']) for r in rows)} transcripts: {base / 'index.html'}"
    )


if __name__ == "__main__":
    main()
