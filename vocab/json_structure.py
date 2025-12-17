import json
import argparse
from typing import Dict, List, Set


# =========================
# 1. 加载停用词
# =========================
def load_stopwords(stopword_file: str) -> Set[str]:
    """
    加载停用词表（每行一个词）
    """
    with open(stopword_file, "r", encoding="utf-8") as f:
        return set(w.strip() for w in f if w.strip())


# =========================
# 2. TinyDB 词表 → JSON
# =========================
def vocab_2_json(
    tinydb_file: str,
    output_file: str,
    stopwords: Set[str] | None = None
) -> Dict[str, str]:
    """
    TinyDB 格式词表 → 普通 JSON {word_id: word}
    同时过滤停用词
    """
    with open(tinydb_file, "r", encoding="utf-8") as f:
        vocab_tiny = json.load(f)

    vocab_new = {}

    for word_id, info in vocab_tiny["_default"].items():
        word = info["word"]

        # 停用词过滤
        if stopwords and word in stopwords:
            continue

        vocab_new[word_id] = word

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab_new, f, ensure_ascii=False, indent=2)

    print(f"[OK] 词表保存到 {output_file}（保留 {len(vocab_new)} 个词）")
    return vocab_new


# =========================
# 3. TinyDB 文档词频 → JSON
# =========================
def freq_2_json(
    tinydb_file: str,
    output_file: str,
    valid_vocab_ids: Set[str]
) -> List[Dict[str, int]]:
    """
    TinyDB 格式文章词频 → 列表形式
    [{word_id: freq, ...}, ...]
    仅保留 vocab 中存在的词
    """
    with open(tinydb_file, "r", encoding="utf-8") as f:
        docs_tiny = json.load(f)

    docs_new = []

    for doc_info in docs_tiny["_default"].values():
        words = doc_info.get("words", {})

        filtered = {
            word_id: freq
            for word_id, freq in words.items()
            if word_id in valid_vocab_ids and freq > 0
        }

        if filtered:
            docs_new.append(filtered)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_new, f, ensure_ascii=False, indent=2)

    print(f"[OK] 文档词频保存到 {output_file}（共 {len(docs_new)} 篇）")
    return docs_new


# =========================
# 4. 主入口
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="TinyDB → LDA/OLDA 可用 JSON（含停用词过滤）"
    )
    parser.add_argument("--vocab_tinydb", required=True, help="TinyDB 词表 JSON")
    parser.add_argument("--doc_tinydb", required=True, help="TinyDB 文档词频 JSON")
    parser.add_argument("--stopwords", required=True, help="停用词 txt 文件")
    parser.add_argument("--out_dir", required=True, help="输出目录")

    args = parser.parse_args()

    vocab_out = f"{args.out_dir}/vocab.json"
    freq_out = f"{args.out_dir}/freq.json"

    # 1. 加载停用词
    stopwords = load_stopwords(args.stopwords)
    print(f"[INFO] 加载停用词 {len(stopwords)} 个")

    # 2. 处理词表
    vocab = vocab_2_json(
        args.vocab_tinydb,
        vocab_out,
        stopwords=stopwords
    )

    # 3. 处理文档词频（严格对齐词表）
    valid_vocab_ids = set(vocab.keys())
    freq_2_json(
        args.doc_tinydb,
        freq_out,
        valid_vocab_ids=valid_vocab_ids
    )

    print("\n🎉 数据准备完成，可直接用于 LDA / OLDA 训练")


if __name__ == "__main__":
    main()