"""Phase A+ (Mem0 v5 一部): attribute の cardinality registry と condition 互換判定。

gpt-5.5-pro 検算(2026-07-03)の指摘を反映:
(3) attribute registry with cardinality — 趣味・アレルギー等の multi/set_like を
    single 扱いすると「追加」を「更新」で消してしまう。attribute ごとに更新ルールを持つ。
(4) condition compatibility — condition は完全一致でなく same/disjoint/subset/superset/
    unknown で判定。disjoint(別条件下)のみ addition、それ以外は conflict 判定へ進める。
"""

# cardinality:
#   single   単一値・新値で上書き (residence, occupation ...)
#   temporal 時系列で上書き        (weight ...)
#   set_like 複数値・追加型         (habit.*, preference.*, possession.* ...)
_ATTRIBUTE_CARDINALITY = {
    "weight": "temporal",
    "height": "single",
    "residence": "single",
    "occupation": "single",
    "affiliation": "single",
    "relationship_status": "single",
    "current_focus": "single",
}
_DOMAIN_CARDINALITY = {
    "habit": "set_like",
    "preference": "set_like",
    "possession": "set_like",
}
_DEFAULT_CARDINALITY = "single"

# 「新しい値が来たら古い値を archive すべき」cardinality。
# set_like は追加型なので archive しない(既存の趣味・好みを消さない)。
_UPDATE_TYPES = {"single", "temporal"}


# v5 T4 (gpt-5.5-pro 指摘5「cross-attribute conflict」): 別 attribute でも
# 意味的に矛盾しうる属性の集合。決定ルートは「同一 attribute or 同一 group」で
# 既存 fact を照会する。初期はほぼ空の枠 — 運用で観測された矛盾ペアを追加する
# (例: {"caffeine": {"habit.coffee", "preference.caffeine"}})。
CONFLICT_GROUPS: dict[str, set[str]] = {}


def get_conflict_group(attribute: str) -> set[str]:
    """attribute が属する conflict group の全メンバー(自分含む)を返す。

    どの group にも属さなければ {attribute} 単独。空 attribute は空 set。
    """
    if not attribute:
        return set()
    a = attribute.lower()
    for members in CONFLICT_GROUPS.values():
        if a in members:
            return set(members)
    return {a}


def get_cardinality(attribute: str) -> str:
    if not attribute:
        return _DEFAULT_CARDINALITY
    a = attribute.lower()
    if a in _ATTRIBUTE_CARDINALITY:
        return _ATTRIBUTE_CARDINALITY[a]
    base = a.split(".", 1)[0].split(":", 1)[0]
    if base in _DOMAIN_CARDINALITY:
        return _DOMAIN_CARDINALITY[base]
    # custom:* は保守的に set_like(誤って追加を上書き消去しない側に倒す)
    if a.startswith("custom:"):
        return "set_like"
    return _DEFAULT_CARDINALITY


def is_update_cardinality(attribute: str) -> bool:
    """single/temporal のみ True。set_like は追加型なので archive しない。"""
    return get_cardinality(attribute) in _UPDATE_TYPES


def condition_compatibility(cond_a: str, cond_b: str) -> str:
    """2つの condition の関係を same/disjoint/subset/superset/unknown で返す。

    - 両方空/片方空       → unknown(判定不能 → conflict 判定へ進める)
    - 完全一致            → same
    - 別軸(time vs context) → unknown(重なりうるので保守的に conflict 判定へ)
    - 同軸で階層関係       → subset/superset (time:morning vs time:morning.weekday)
    - 同軸で異なる値       → disjoint (time:morning vs time:night)
    """
    a = (cond_a or "").strip().lower()
    b = (cond_b or "").strip().lower()
    if not a or not b:
        return "unknown"
    if a == b:
        return "same"
    pa = a.split(":", 1)[0]
    pb = b.split(":", 1)[0]
    if pa != pb:
        return "unknown"
    va = a.split(":", 1)[1] if ":" in a else a
    vb = b.split(":", 1)[1] if ":" in b else b
    if va.startswith(vb + "."):
        return "subset"     # a は b の下位 (morning.weekday ⊂ morning)
    if vb.startswith(va + "."):
        return "superset"
    return "disjoint"       # 同軸で無関係 (morning vs night)


def conditions_are_conflict_candidate(cond_a: str, cond_b: str) -> bool:
    """conflict 判定へ進めるべきか。disjoint(別条件下)のみ False(=addition)。

    same/subset/superset/unknown は「同じ条件 or 重なりうる」ので、更新/訂正の
    可能性があり conflict 判定(classify_conflict)へ進める。
    """
    return condition_compatibility(cond_a, cond_b) != "disjoint"
