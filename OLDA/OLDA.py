import json
import argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from typing import Dict


def build_corpus(vocab_json: str, doc_freq_json: str):
  
    with open(vocab_json, "r", encoding="utf-8") as f:
        id2word_raw = json.load(f)   # { "1": "万人", ... }
    token2id = {word: int(idx) for idx, word in id2word_raw.items()}
    dictionary = corpora.Dictionary()
    dictionary.token2id = token2id
    dictionary.id2token = {v: k for k, v in token2id.items()}
    dictionary.num_docs = 0
    with open(doc_freq_json, "r", encoding="utf-8") as f:
        docs = json.load(f)
    corpus = []
    for doc in docs:
        bow = []
        for word, freq in doc.items():
            if word in token2id:
                bow.append((token2id[word], freq))
        corpus.append(bow)

    return corpus, dictionary
def train_lda(corpus, dictionary, num_topics=10, passes=10, iterations=50, chunksize=100):
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        alpha='auto',
        eta='auto',
        chunksize=chunksize
    )
    return lda_model


def extract_topics(lda_model, num_topics, top_words=10):
    topics = {}
    for i, topic in lda_model.show_topics(num_topics=num_topics, num_words=top_words, formatted=False):
        topics[i] = [word for word, prob in topic]
    return topics


def get_document_topics(lda_model, corpus):
    return [lda_model.get_document_topics(doc_bow) for doc_bow in corpus]

def save_json(obj, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLDA 训练脚本")
    parser.add_argument("--vocab", type=str, required=True, help="词表 JSON 文件路径")
    parser.add_argument("--docs", type=str, required=True, help="文章词频 JSON 文件路径")
    parser.add_argument("--num_topics", type=int, default=10, help="主题数量")
    parser.add_argument("--passes", type=int, default=10, help="LDA passes")
    parser.add_argument("--iterations", type=int, default=50, help="LDA iterations")
    parser.add_argument("--chunksize", type=int, default=100, help="LDA chunksize")
    parser.add_argument("--top_words", type=int, default=15, help="每个主题输出的词数")
    parser.add_argument("--topics_out", type=str, default="topics.json", help="主题词输出文件")
    parser.add_argument("--doc_topics_out", type=str, default="doc_topics.json", help="文档主题分布输出文件")

    args = parser.parse_args()

    # 加载词表和语料
   
    corpus, dictionary = build_corpus(args.vocab,args.docs)

    # 训练 LDA
    lda_model = train_lda(
        corpus=corpus,
        dictionary=dictionary,
        num_topics=args.num_topics,
        passes=args.passes,
        iterations=args.iterations,
        chunksize=args.chunksize
    )

    # 输出主题词
    topics = extract_topics(lda_model, num_topics=args.num_topics, top_words=args.top_words)
    save_json(topics, args.topics_out)

    # 输出文档主题分布
    doc_topics = get_document_topics(lda_model, corpus)
    save_json(doc_topics, args.doc_topics_out)

    print(f"OLDA 训练完成，主题词已保存到 {args.topics_out}，文档主题分布已保存到 {args.doc_topics_out}")