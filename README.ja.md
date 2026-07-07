# Mem0α

[English](README.md) | 日本語

LLMエージェントに長期記憶を持たせるためのライブラリです。プロバイダには依存しません。[`mem0`](https://github.com/mem0ai/mem0) の上に薄く乗せる形で、会話エージェントが実運用で欲しくなる検索まわり・保存まわりの作り込みを足しています。認証層は差し替え可能で、**すでにある Claude Code CLI のログインをそのまま使える**ので、多くの場合 API キーの用意すら要りません。

> パッケージ名は `mem0-alpha`、import 名は `llm_mem0` です。

```python
from llm_mem0 import add_memories, search_memories, format_memories_for_prompt

# 会話1ターンから「ユーザー本人の事実」だけを抜き出して保存する
await add_memories(
    user_text="ベルリンに引っ越して、Rust を始めた",
    assistant_text="いいですね。borrow checker には慣れましたか？",
    user_id="alice",
)

# あとで、別のクエリに関連する記憶を引く。そのままプロンプトに差し込める形で返る
memories = await search_memories("このユーザーはどこに住んでる？", user_id="alice")
print(format_memories_for_prompt(memories))
# [Long-term memory — facts about this user]
# - ベルリンに住んでいる [imp=4 Hot] [2026-07-08]
# - Rust を学習中 [imp=3 Hot] [2026-07-08]
# [End of Long-term memory]
```

## `mem0` をそのまま使うのとの違い

`mem0` はベクトルストア周りの面倒を引き受けてくれます。Mem0α はその上に、長く使い続ける個人向けエージェントで必要になるものを重ねます。

- **抽出ゲート** — 小型モデルを1回呼ぶだけで、生の会話ターンから「ユーザー本人の事実」だけをきれいに取り出します。アシスタントの発言が混ざり込むことはなく、保存前に検証もします。
- **2段階の重複排除** — まず cosine で候補を絞り、次に LLM がマージの可否を判断します。さらに属性の語彙を統制しているので、新しい値（体重・住所・職業など）が古い値をきちんと置き換えます。同じ事実が言い換えのたびに積み上がることはありません。
- **ハイブリッド検索** — ベクトル検索と BM25 のキーワード検索を RRF で束ね、必要ならクエリ書き換えと HyDE で広げてから並べ替えます。人名や曲名のような固有名詞が埋め込み空間で埋もれて取りこぼす、という事故を防げます。
- **エンティティ／関係グラフ** — 別名・共起・1ホップ展開を扱います。
- **インジェクション対策の整形** — 取り出したテキストは、プロンプトに戻す前に、骨組みを偽装するようなマーカーを取り除きます。

設計の詳細は [ARCHITECTURE.ja.md](ARCHITECTURE.ja.md) にあります。

## 認証 — CLI のログインを流用してもいいし、API キーでもいい

`llm_mem0` は次の順で認証方法を自動判定します。

1. **Claude Code CLI のセッション** — `claude` コマンドが保持している OAuth セッションをそのまま借ります。だから **`ANTHROPIC_API_KEY` を別途用意する必要がありません**。トークンを読み、必須の `anthropic-beta: oauth-2025-04-20` ヘッダを付け、Anthropic SDK が Bearer 認証を使うようパッチを当てます。トークンの置き場所は OS で違い、**Linux** は `~/.claude/.credentials.json`、**macOS** は CLI がログイン **Keychain** に入れているのでそこから読みます（読めなければファイルにフォールバック）。あくまで **あなた自身の**ログインを使い回すだけで、CLI のネットワーク指紋を偽装するようなことはしません。

   **トークンの更新は保守的に扱い、CLI のログインを壊さないことを最優先にしています。** トークンが切れていたら、まず CLI 自身の保存先（Keychain／ファイル）を読み直します。CLI が動いていればトークンを新しく保ってくれているので、こちらは何もローテーションせずにそれへ相乗りできます。それでもまだ切れている場合の動きは、トークンの出どころで変わります。**ファイル**由来なら、こちらで更新して CLI が読むのと同じファイルに書き戻します（両者で整合が取れる）。**Keychain**由来のときは、既定では更新しません。更新するとトークンがローテーションされますが、新しいトークンを Keychain に確実に書き戻す手段がなく、そのままだと CLI 側のログインを無効にしてしまうからです。この場合ライブラリは「`claude` を一度実行すればトークンが更新される」旨をログに出し、その呼び出しでは記憶を返しません。それでも直接更新したいなら `LLM_MEM0_ALLOW_TOKEN_REFRESH=1` を設定します（`claude` の再ログインが必要になることを承知の上で）。
2. **標準の API キー** — 上記が見つからなければ、環境変数の API キーに切り替わります。`mem0` が対応しているプロバイダならどれでも動きます。

```bash
# ケース1: すでに Claude Code にログイン済みなら、設定なしでそのまま動く

# ケース2: API キーで OpenAI（など任意のプロバイダ）を使う
export MEM0_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MEM0_LLM_MODEL=gpt-5-mini
```

> Codex CLI のローカル認証キャッシュ（`~/.codex/auth.json`）は、あえて使いません。OpenAI がこれを「内部実装であり、外部ツールからの読み取りはサポートしない」と明言しているためです。OpenAI／Codex のモデルは、標準の `OPENAI_API_KEY` 経由で使ってください。

## インストール

```bash
pip install llm-mem0
# Anthropic SDK も入れる（Claude Code CLI 流用 or Anthropic の API キー用）
pip install "llm-mem0[anthropic]"
# OpenAI SDK も入れる
pip install "llm-mem0[openai]"
# BM25 インデックス用の日本語トークナイザ（任意）
pip install "llm-mem0[japanese]"
```

## 動かす前に

- **ベクトルストア。** 既定では ChromaDB の HTTP サーバーにつなぎます。
  ```bash
  chroma run --host 127.0.0.1 --port 8765
  ```
  サーバーを立てたくなければ `CHROMA_MODE=embedded` にすると、`~/.llm_mem0/state` の下にファイルとして持ちます（単一プロセス専用）。
- **埋め込みモデル。** 既定は OpenAI の `text-embedding-3-small` で、`OPENAI_API_KEY` が要ります。`MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL` で差し替えられます（たとえばローカルの HuggingFace モデル）。

## 公開 API

| 関数 | 何をするか |
|---|---|
| `add_memories(user_text=, assistant_text=, user_id=, …)` | 会話ターンから本人の事実を抽出して保存する |
| `search_memories(query, user_id, limit=8)` | 関連度でフィルタしつつ意味検索する |
| `search_memories_multi(queries, user_id, …)` | 複数クエリに広げて検索する |
| `search_memories_smart(query, user_id, rewrite=True, rerank=True)` | クエリ書き換え + HyDE + 並べ替え付きの検索 |
| `should_use_memory_llm_mode(...)` | smart 検索にコストをかけるべきか判断するヒューリスティック |
| `format_memories_for_prompt(memories)` | 記憶をインジェクション対策済みのプロンプト断片に整形する |
| `format_history_for_prompt(hits)` | 履歴検索の結果をプロンプト断片に整形する |
| `extract_facts_for_self(user_text, assistant_text, …)` | 抽出だけを行う（事実のみを返す） |
| `get_all_memories(user_id, limit=None)` / `delete_memory(memory_id)` | 低レベルの保守用 |

## 設定

設定はすべて環境変数で、安全な既定値が入っています（実体は `llm_mem0/settings.py`）。よく使うものは次のとおりです。

| 環境変数 | 既定値 | 意味 |
|---|---|---|
| `CHROMA_MODE` | `server` | `server` か `embedded` |
| `CHROMA_HOST` / `CHROMA_PORT` | `127.0.0.1` / `8765` | Chroma サーバーのアドレス |
| `MEM0_COLLECTION_NAME` | `memories` | Chroma のコレクション名 |
| `MEM0_LLM_PROVIDER` | 自動判定 | `anthropic` / `openai` など（API キー使用時） |
| `MEM0_LLM_MODEL` | バックエンド既定 | 抽出・重複排除・並べ替えに使うモデル |
| `MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL` | `openai` / `text-embedding-3-small` | 埋め込みモデル |
| `LLM_MEM0_STATE_DIR` | `~/.llm_mem0/state` | sqlite インデックスと embedded Chroma の置き場所 |
| `MEM0_HYBRID_ENABLED` | `true` | BM25 とベクトル検索を束ねる |
| `MEM0_HYDE_ENABLED` | `true` | HyDE（仮の解答文）でクエリを広げる |

## サンプル

- [`examples/basic_usage.py`](examples/basic_usage.py) — Claude Code CLI のセッションを流用する例
- [`examples/with_openai.py`](examples/with_openai.py) — OpenAI の API キーで動かす例

## ライセンス

MIT。[LICENSE](LICENSE) を参照してください。
