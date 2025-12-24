import json
import math
import argparse
from typing import Dict, List, Set


# ======================
# IO
# ======================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ======================
# 清洗 freq.json
# ======================

def normalize_docs(docs):
    """
    保证 docs -> List[Dict[str, int]]
    """
    fixed = []
    bad = 0

    for doc in docs:
        if isinstance(doc, dict):
            fixed.append(doc)
        elif isinstance(doc, str):
            try:
                d = json.loads(doc)
                if isinstance(d, dict):
                    fixed.append(d)
                else:
                    bad += 1
            except Exception:
                bad += 1
        else:
            bad += 1

    print(f"[INFO] freq 清洗完成：保留 {len(fixed)} 篇，丢弃 {bad} 篇异常文档")
    return fixed


# ======================
# DF / IDF / TF-IDF
# ======================

def compute_df(docs: List[Dict[str, int]]) -> Dict[str, int]:
    df = {}
    for doc in docs:
        for wid in doc:
            df[wid] = df.get(wid, 0) + 1
    return df


def compute_idf(df: Dict[str, int], doc_count: int) -> Dict[str, float]:
    return {
        wid: math.log((doc_count + 1) / (c + 1)) + 1
        for wid, c in df.items()
    }


def compute_tfidf(
    docs: List[Dict[str, int]],
    idf: Dict[str, float]
) -> Dict[str, float]:
    """
    词级 TF-IDF（全语料累计）
    """
    tfidf = {}

    for doc in docs:
        doc_len = sum(doc.values())
        if doc_len == 0:
            continue

        for wid, tf in doc.items():
            tf_norm = tf / doc_len
            tfidf[wid] = tfidf.get(wid, 0.0) + tf_norm * idf.get(wid, 0.0)

    return tfidf


# ======================
# TF-IDF + DF 过滤 + 压缩到目标词表规模
# ======================

def select_vocab(
    tfidf: Dict[str, float],
    df: Dict[str, int],
    doc_count: int,
    min_df: int,
    max_df_ratio: float,
    min_tfidf: float,
    target_min: int,
    target_max: int
) -> List[str]:
    """
    返回：按 TF-IDF 排序后的 old_id 列表
    """

    # 规则过滤
    candidates = [
        wid for wid, score in tfidf.items()
        if df.get(wid, 0) >= min_df
        and df.get(wid, 0) <= doc_count * max_df_ratio
        and score >= min_tfidf
    ]

    print(f"[INFO] 规则过滤后词数: {len(candidates)}")

    # 按 TF-IDF 排序
    candidates.sort(key=lambda w: tfidf[w], reverse=True)

    # 强制压缩到目标区间
    if len(candidates) > target_max:
        candidates = candidates[:target_max]
    elif len(candidates) < target_min:
        print("[WARN] 词表小于目标下限，请检查阈值")

    return candidates


# ======================
# 直接重排 vocab / freq（不输出 id_map）
# ======================

def rebuild_vocab_and_docs(
    vocab: Dict[str, str],
    docs: List[Dict[str, int]],
    ordered_old_ids: List[str]
):
    """
    old_id -> new_id（0..N-1）
    """

    old_to_new = {
        old_id: str(new_id)
        for new_id, old_id in enumerate(ordered_old_ids)
    }

    # 新 vocab
    new_vocab = {
        str(new_id): vocab[old_id]
        for new_id, old_id in enumerate(ordered_old_ids)
    }

    # 新 docs
    new_docs = []
    for doc in docs:
        new_doc = {}
        for old_id, tf in doc.items():
            if old_id in old_to_new:
                new_doc[old_to_new[old_id]] = tf
        if new_doc:
            new_docs.append(new_doc)

    return new_vocab, new_docs


# ======================
# main
# ======================

def main():
    parser = argparse.ArgumentParser(
        description="TF-IDF 词表压缩 + 直接重编号（OLDA 专用）"
    )

    parser.add_argument("--vocab", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--min_df", type=int, default=10)
    parser.add_argument("--max_df_ratio", type=float, default=0.4)
    parser.add_argument("--min_tfidf", type=float, default=0.02)

    parser.add_argument("--target_min", type=int, default=10000)
    parser.add_argument("--target_max", type=int, default=12000)

    args = parser.parse_args()

    vocab = load_json(args.vocab)
    docs_raw = load_json(args.docs)

    if not isinstance(vocab, dict):
        raise TypeError("vocab.json 必须是 dict {id: word}")

    docs = normalize_docs(docs_raw)
    doc_count = len(docs)

    print(f"[INFO] 文档数: {doc_count}")
    print(f"[INFO] 原词表大小: {len(vocab)}")

    df = compute_df(docs)
    idf = compute_idf(df, doc_count)
    tfidf = compute_tfidf(docs, idf)

    ordered_old_ids = select_vocab(
        tfidf=tfidf,
        df=df,
        doc_count=doc_count,
        min_df=args.min_df,
        max_df_ratio=args.max_df_ratio,
        min_tfidf=args.min_tfidf,
        target_min=args.target_min,
        target_max=args.target_max
    )

    new_vocab, new_docs = rebuild_vocab_and_docs(
        vocab, docs, ordered_old_ids
    )

    save_json(new_vocab, f"{args.out_dir}/vocab_tfidf.json")
    save_json(new_docs, f"{args.out_dir}/freq_tfidf.json")

    print(f"[OK] 最终词表大小: {len(new_vocab)}")
    print(f"[OK] 最终文档数: {len(new_docs)}")


if __name__ == "__main__":
    main()
