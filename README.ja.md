# Mem0α

[English](README.md) | 日本語

プロバイダ非依存の **LLMエージェント向け長期記憶ライブラリ** です。[`mem0`](https://github.com/mem0ai/mem0) の薄いラッパーとして、会話エージェントが実際に必要とする「検索品質」「取り込み品質」の仕組みを追加し、さらに **既存の Claude Code CLI セッションをそのまま流用** できる着脱式の認証層を備えています。別途 API キーを用意しなくても動きます。

> ディストリビューション名は `mem0-alpha`、import 名は `llm_mem0` です。

```python
from llm_mem0 import add_memories, search_memories, format_memories_for_prompt

# 会話ターンから「ユーザー本人に関するクリーンな事実」だけを抽出して保存する。
await add_memories(
    user_text="ベルリンに引っ越して Rust を始めた。",
    assistant_text="いいですね！ borrow checker はどうですか？",
    user_id="alice",
)

# あとで — 新しいクエリに関連する記憶を取り出し、そのままプロンプトに差し込む。
memories = await search_memories("このユーザーはどこに住んでる？", user_id="alice")
print(format_memories_for_prompt(memories))
# [Long-term memory — facts about this user]
# - ベルリンに住んでいる [imp=4 Hot] [2026-07-08]
# - Rust を学習中 [imp=3 Hot] [2026-07-08]
# [End of Long-term memory]
```

## なぜ `mem0` を直接使わないのか

`mem0` はベクトルストアまわりの面倒を引き受けてくれます。このライブラリはその上に、長期運用の個人エージェントに必要なものを足します。

- **自己事実の抽出ゲート** — 小型モデルへの1回の呼び出しで、生の会話ターンを *ユーザー本人に関する* クリーンな事実へ変換（アシスタント発話からの話者混入なし）。保存前に検証します。
- **2段階 dedup** — cosine による事前フィルタ → LLM によるマージ判定。加えて統制語彙（controlled vocabulary）により、新しい値（体重・居住地・職業…）が古い値を決定的に上書き（archive）し、重複が積み上がるのを防ぎます。
- **ハイブリッド検索** — ベクトル検索と BM25 キーワードインデックスを Reciprocal Rank Fusion (RRF) で融合。さらに query rewrite + HyDE で拡張し rerank するので、固有名詞（人名・曲名など）が埋め込み空間で埋もれても取りこぼしません。
- **エンティティ/関係グラフ** — エイリアス、共起、1-hop 展開。
- **インジェクション対策の整形** — 取り出したテキストは、プロンプトに再投入する前にプロンプト骨格マーカーをサニタイズします。

設計の全体像は [ARCHITECTURE.ja.md](ARCHITECTURE.ja.md) を参照してください。

## 認証 — CLI セッション流用でも、任意の API キーでも

`llm_mem0` は以下の順で認証バックエンドを自動検出します。

1. **Claude Code CLI セッション** — `claude` CLI が保持している OAuth セッションを流用するので、**別途 `ANTHROPIC_API_KEY` は不要**です。トークンを読み取り、必須の `anthropic-beta: oauth-2025-04-20` ヘッダを付与し、Anthropic SDK を Bearer 認証に切り替えるモンキーパッチを当てます。**Linux** ではトークンは `~/.claude/.credentials.json` から、**macOS** では CLI がログイン **Keychain** に保存しているのでそこから読み取ります（ファイルへフォールバック）。これは *あなた自身の* 認証済みセッションの流用であり、CLI のネットワーク指紋を偽装するものではありません。

   **トークンの鮮度管理は保守的で、CLI のログインを絶対に壊さない設計です。** トークンが失効している場合、まず CLI 自身のストア（Keychain / ファイル）を再読み込みします（CLI が通常利用で鮮度を保っているため）。それでも失効していた場合の挙動はソース次第です。**ファイル** 由来なら直接 refresh して、CLI が読む同じファイルに書き戻します（同期が取れる）。一方 **Keychain** 由来のトークンはデフォルトでは refresh しません。refresh はトークンをローテーションしますが、新トークンを Keychain へ確実に書き戻せないため、それをやると CLI 側のログインを無効化してしまうからです。この場合ライブラリは「`claude` コマンドを1回叩けばトークンが更新される」旨をログに出し、その呼び出しでは記憶を返しません。それでも直接 refresh したい場合は `LLM_MEM0_ALLOW_TOKEN_REFRESH=1`（`claude` 再ログインが必要になる可能性を許容）を設定します。
2. **標準 API キー** — 上記が無ければ、環境変数の API キーにフォールバックします。`mem0` が対応する任意のプロバイダで動きます。

```bash
# パターン1: すでに Claude Code にログイン済み? 何も設定しなくてそのまま動きます。

# パターン2: API キーで OpenAI（や任意のプロバイダ）を使う
export MEM0_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MEM0_LLM_MODEL=gpt-5-mini
```

> Codex CLI のローカル認証キャッシュ（`~/.codex/auth.json`）は **意図的に流用していません**。OpenAI が公式に「内部実装の詳細でありサードパーティによる読み取りはサポートしない」と明記しているためです。OpenAI/Codex のモデルは標準の `OPENAI_API_KEY` 経由で利用してください。

## インストール

```bash
pip install llm-mem0
# Anthropic SDK 付き（Claude Code CLI 流用 or Anthropic API キー用）:
pip install "llm-mem0[anthropic]"
# OpenAI SDK 付き:
pip install "llm-mem0[openai]"
# BM25 インデックス用の日本語トークナイザ（任意）:
pip install "llm-mem0[japanese]"
```

## 前提

- **ベクトルストア。** デフォルトで `llm_mem0` は ChromaDB の HTTP サーバーと通信します:
  ```bash
  chroma run --host 127.0.0.1 --port 8765
  ```
  もしくは `CHROMA_MODE=embedded` で `~/.llm_mem0/state` 配下のオンディスクストアを使います（単一プロセス限定）。
- **埋め込み器（embedder）。** デフォルトは OpenAI `text-embedding-3-small`（`OPENAI_API_KEY` が必要）。`MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL` で差し替え可能です（例: ローカルの HuggingFace embedder）。

## 公開 API

| 関数 | 役割 |
|---|---|
| `add_memories(user_text=, assistant_text=, user_id=, …)` | 会話ターンから自己事実を抽出して保存する。 |
| `search_memories(query, user_id, limit=8)` | 関連度距離ゲート付きの意味検索。 |
| `search_memories_multi(queries, user_id, …)` | 複数クエリで検索を扇状に展開する。 |
| `search_memories_smart(query, user_id, rewrite=True, rerank=True)` | query rewrite + HyDE + rerank 付きの検索。 |
| `should_use_memory_llm_mode(...)` | smart パスにコストをかけるか判断するヒューリスティックゲート。 |
| `format_memories_for_prompt(memories)` | 記憶をインジェクション対策済みのプロンプトブロックに整形する。 |
| `format_history_for_prompt(hits)` | 履歴検索ヒットをプロンプトブロックに整形する。 |
| `extract_facts_for_self(user_text, assistant_text, …)` | 抽出ステップだけ（事実のみ）。 |
| `get_all_memories(user_id, limit=None)` / `delete_memory(memory_id)` | 低レベルの保守用。 |

## 設定

すべての設定は安全なデフォルト付きの環境変数です（`llm_mem0/settings.py` に集約）。よく使うもの:

| 環境変数 | デフォルト | 意味 |
|---|---|---|
| `CHROMA_MODE` | `server` | `server` または `embedded` |
| `CHROMA_HOST` / `CHROMA_PORT` | `127.0.0.1` / `8765` | Chroma サーバーのアドレス |
| `MEM0_COLLECTION_NAME` | `memories` | Chroma コレクション名 |
| `MEM0_LLM_PROVIDER` | 自動 | `anthropic` / `openai` / …（API キーモード用） |
| `MEM0_LLM_MODEL` | バックエンド既定 | 抽出/dedup/rerank に使うモデル |
| `MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL` | `openai` / `text-embedding-3-small` | 埋め込み器 |
| `LLM_MEM0_STATE_DIR` | `~/.llm_mem0/state` | sqlite インデックス + embedded Chroma |
| `MEM0_HYBRID_ENABLED` | `true` | BM25 とベクトル検索を融合する |
| `MEM0_HYDE_ENABLED` | `true` | 仮想解答（HyDE）によるクエリ拡張 |

## サンプル

- [`examples/basic_usage.py`](examples/basic_usage.py) — Claude Code CLI セッションを流用する例。
- [`examples/with_openai.py`](examples/with_openai.py) — OpenAI API キーで動かす例。

## ライセンス

MIT — [LICENSE](LICENSE) を参照。
