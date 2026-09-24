# gpt-live-eval-ja-retail

OpenAI Cookbookの[GPT-Live evaluation guide](https://developers.openai.com/cookbook/examples/audio/voice_agent_evaluation)のハーネスを、日本語の小売サポートに適用した実験コードです。
題材はSB Intuitionsの[J-tau-bench](https://github.com/sbintuitions/j-tau-bench)のretailドメインです。
結果と所感は[記事](docs/blog-overview.md)にまとめています。

## 構成

| パス | 内容 |
|---|---|
| `voice_eval/` | 実験用のCLIと、retail用のアダプタ。ツール実行ワーカー、採点、監査を含む |
| `prompts/retail/` | オペレーターのプロンプト。variantごとの差分は各フォルダの`CHANGE.md`に記録 |
| `data/retail/` | シナリオ、チェックポイント、実験用の架空の本人確認データ |
| `vendor/gpt_live_evals/` | OpenAIの評価ハーネス。出典は`UPSTREAM.json` |
| `vendor/j_tau/` | J-tau-benchの業務・データ・評価ソース。固定コミットは`data/retail/source.json` |
| `tests/` | APIを使わないテスト |

## セットアップ

Python 3.12とuvを使います。J-tau-benchの依存関係は専用の環境に分けています。

```bash
uv sync --frozen --python 3.12
uv sync --project vendor/j_tau --frozen --python 3.12
cp .env.example .env   # OPENAI_API_KEY と GEMINI_API_KEY を設定
uv run python -m voice_eval doctor --domain retail
uv run pytest -q tests
```

`.env`はGitで管理しません。キーは入力スナップショットやマニフェストにも保存しません。

## 記事の実験を再現する

実行するとAPI利用料が発生します。

記事のRUNは、variant `name-kana`で6ケースを1回ずつ実行したものです。
このvariantでは、本人確認の氏名を読みがなで照合し、一文字ずつの確認はさせません。1回の会話の上限は10分です。

```bash
uv run python -m voice_eval run run --domain retail --suite pilot --variant name-kana --new-attempt
```

CRAWL/WALKは、Gemini TTSで入力音声を作ってから実行します。

```bash
uv run python -m voice_eval prepare --domain retail
uv run python -m voice_eval run crawl --domain retail --suite pilot
uv run python -m voice_eval run walk --domain retail --suite pilot
```

結果は`artifacts/retail/name-phone-v1/live/<mode>/<variant>/<task>/attempt-N/`に保存されます。`result.json`が各試行の評価結果です。
既存の試行は上書きしません。同じ条件を繰り返すときは`--new-attempt`を付けます。

## 注意

- 公式のJ-tau-benchと同じ条件のスコアではありません。本人確認を電話窓口向けに置き換え、音声向けの運用ルールを補い、採点モデルも変えています。
- 本人確認に使う電話番号は、実験用に付与した架空の番号です。
- モデルの利用権はOpenAIのアカウントに依存します。

## ライセンス

- `vendor/`以外のコードと文書: [MIT License](LICENSE)
- `vendor/gpt_live_evals/`: MIT License, Copyright (c) 2025 OpenAI。[原文](vendor/gpt_live_evals/LICENSE)
- `vendor/j_tau/`と、そこから作った`data/retail/`: Modified-MIT License, Copyright (c) 2026 SB Intuitions Corp.。商用目的でAI・機械学習モデルの学習に使うことは禁止されています。[原文](vendor/j_tau/LICENSE)
