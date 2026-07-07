# アーキテクチャ

[English](ARCHITECTURE.md) | 日本語

`llm_mem0` は [`mem0`](https://github.com/mem0ai/mem0) の薄い、意見の強い（opinionated）ラッパーです。`mem0` はベクトルストアまわりの難しい部分（埋め込み、upsert、類似検索）を既に引き受けており、マルチプロバイダ対応です。このライブラリは、その素のベクトルストアの上に、会話エージェントが必要とする検索・取り込み品質の仕組みを足し、さらに「LLM の CLI にログイン済みなら別途 API キーを用意しなくていい」着脱式の認証層を追加します。

## なぜ mem0 を直接使わないのか

`mem0` の組み込み抽出器は、簡潔な英語の `Prefers ~` 形式の文を生成し、1ターンにユーザー発話とアシスタント発話の両方が含まれると話者が混ざります。長期運用の個人エージェントでは、これが3つの問題を生みました。本ライブラリはそれを解消します。

1. **話者混入（Speaker bleed）** — アシスタントや他者に関する事実がユーザーの記憶に漏れ込む。ここでは抽出を *自己帰属可能な*（`speaker == "self"`）事実に限定し、保存前に検証します。
2. **dedup 肥大** — 抽出器が毎ターン同じ事実を言い換えるため、ニアデュープが積み上がる。ここでは2段階 dedup（cosine 事前フィルタ → LLM マージ判定）で畳み込み、属性レジストリが単一値属性（体重・居住地…）を決定的に上書きします。
3. **想起ミス** — 純粋なベクトル検索では、無関係な近傍に埋め込まれた固有名詞（人名・曲名）を取りこぼす。ここでは検索をハイブリッド化し、ベクトル検索と BM25 キーワードインデックスを Reciprocal Rank Fusion で融合、必要に応じて query rewrite + HyDE で拡張し rerank します。

## モジュール構成

| モジュール | 責務 |
|---|---|
| `auth/` | 着脱式の認証バックエンド（下記参照）。 |
| `client.py` | `mem0.Memory` のシングルトン + 設定ビルダ。低レベルの `get_all_memories` / `delete_memory`。 |
| `settings.py` | 環境変数ベースの設定デフォルトを一箇所に集約。 |
| `ingest.py` | `add_memories` — 取り込みの入口（抽出ゲート、hash dedup、ニアデュープマージ、graph/BM25/fact-store インデックス化）。 |
| `extract.py` | 小型モデルによる、会話ターンからの自己事実 + エンティティ関係の抽出。taxonomy/importance/tier 分類。 |
| `dedup.py` | 2段階 dedup: cosine 候補ゲート + LLM マージ/衝突判定。 |
| `conflict.py` | 新しい事実が既存を「更新/矛盾/共存」のどれに当たるか分類する。 |
| `attribute_registry.py` | 単一値属性の統制語彙。新しい値が古い値を決定的に archive する。 |
| `reinforce.py` | 繰り返し出る事実の read-modify-write 強化（言及回数、最終確認）。 |
| `search.py` | `search_memories` / `search_memories_multi` / `search_memories_smart` — ハイブリッド検索、query rewrite、HyDE、rerank、graph 展開。 |
| `graph.py` | SQLite のエンティティ/関係グラフ（エイリアス、共起、1-hop 展開）。 |
| `bm25_index.py` | 事実に対する SQLite FTS5 キーワードインデックス（CJK は `morpho` で事前トークナイズ）。 |
| `history_index.py` / `history_embed.py` | 会話全履歴のキーワード/埋め込みインデックス。整理済みの fact store とは別枠。 |
| `morpho.py` | FTS インデックス用の日本語形態素トークナイズ（任意）。トークナイザ未導入時は優雅にフォールバック。 |
| `fact_store.py` | 属性値変化のイベントソーシング SQLite ログ（任意）。 |
| `format.py` | 取り出した記憶/履歴ヒットをプロンプトブロックに整形する。 |
| `sanitize.py` | 未信頼テキストのプロンプトインジェクション対策（sentinel / power-phrase のサニタイズ）。 |
| `batch.py` | 会話ごとのターン蓄積器（バッチ取り込み用）。 |

## 認証層

このライブラリで唯一、本当に新規性のある部分です。LLM の呼び出しは2箇所で発生し、認証バックエンドは単一の資格情報からその両方を賄います。

1. **mem0 自身の内部呼び出し**（`Memory.add()` / `search()` 内での抽出器/埋め込み器）— バックエンドが `mem0_llm_config()` を提供し、`Memory.from_config()` が消費する `{"provider", "config"}` ブロックを返します。
2. **本ライブラリ自身の直接ヘルパー呼び出し**（自己事実抽出、メタデータ分類、dedup 判定、query rewrite/rerank）— バックエンドが `complete(system, user_message, model, max_tokens) -> str` を提供します。プロバイダ非依存の単発補完で、失敗時は `""` を返すため、呼び出し側が SDK ごとに特別扱いする必要がありません。

### バックエンド（この順で自動検出）

- **`AnthropicCliAuth`** — 既存の **Claude Code CLI** の OAuth セッションを流用します。資格情報のソースはプラットフォーム依存で、Linux では `~/.claude/.credentials.json`（`claude` CLI が読み書き・refresh する同じファイル）を読み、macOS では CLI がログイン **Keychain** に保存しているので `security find-generic-password` 経由で読み取ります（ファイルへフォールバック）。必須の `anthropic-beta: oauth-2025-04-20` ヘッダを付与し（これが無い OAuth トークンは 401 で弾かれます）、`anthropic.Anthropic.__init__` にモンキーパッチを当てて、mem0 が内部生成するクライアントが `x-api-key` ではなく Bearer 認証を使うようにします。パッチはスレッドセーフな double-checked locking で保護されています。**これはあなた自身の認証済みセッションの流用であり、CLI のネットワーク指紋を偽装しません** — リクエストは素の Anthropic SDK 経由で出ます。

  **トークンの鮮度 / CLI との共存。** キャッシュしたトークンが失効している場合、バックエンドはまず CLI 自身のストア（Keychain/ファイル）を *再読み込み* します。CLI が稼働していればトークンを新鮮に保っているので、こちらは何もローテーションせずその refresh サイクルに便乗できます。それでも失効している場合にのみ直接 refresh しますが、その際は警告をログに出します。直接 refresh は refresh token をローテーションし、CLI が保持するコピーを無効化して `claude` の再ログインを強いる可能性があるためです。この順序により、ライブラリがユーザーの CLI ログインを黙って壊すことを防いでいます。
- **`ApiKeyAuth`** — 環境変数の標準 API キー（`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、…）。これが、mem0 が内部呼び出しで対応する任意のプロバイダで本ライブラリを動かせる経路です。`complete()` は Anthropic と OpenAI についてネイティブ実装されています。

選択ロジックは `auth/__init__.py:get_auth_backend()`（プロセス全体でキャッシュされるシングルトン）にあります。プロバイダを追加するには `AuthBackend`（`auth/base.py`）を実装し、そこに登録します。

> **スコープ注記。** Codex CLI のローカル認証キャッシュ（`~/.codex/auth.json`）は **意図的に流用していません**。OpenAI が公式に「内部実装の詳細でありサードパーティによる読み取り経路はサポートしない」と明記しており、スキーマも非公開のためです。OpenAI/Codex のモデルは標準の `OPENAI_API_KEY` 経由で使ってください。

## ストレージ

- **ベクトル**: ChromaDB。HTTP サーバー（`CHROMA_MODE=server`、デフォルト。embedded な HNSW インデックスを破損させうる複数プロセス書き込みレースを避けられる）か、embedded なオンディスクストア（`CHROMA_MODE=embedded`）。
- **それ以外すべて**（エンティティグラフ、BM25 インデックス、履歴インデックス、fact store）: `LLM_MEM0_STATE_DIR`（デフォルト `~/.llm_mem0/state`）配下の SQLite ファイル。各インデックスを独立して再構築・破棄できるよう、それぞれ別ファイルにしています。

## 設定

すべてのつまみは安全なデフォルト付きの環境変数で、`settings.py` に集約されています。主なもの: `CHROMA_MODE` / `CHROMA_HOST` / `CHROMA_PORT`、`MEM0_COLLECTION_NAME`、`MEM0_LLM_PROVIDER` / `MEM0_LLM_MODEL`、`MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL`、および検索の機能フラグ（`MEM0_HYBRID_ENABLED`、`MEM0_HYDE_ENABLED`、`MEM0_SCOPE_FILTER_ENABLED`、…）。
