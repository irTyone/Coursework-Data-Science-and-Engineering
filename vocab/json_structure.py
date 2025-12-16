import json
from typing import Dict, List
from core.config import VOCAB_PATH,CONTENT_INFO,DATA_PATH
def vocab_2_json(tinydb_file: str, output_file: str) -> Dict[str, str]:

    with open(tinydb_file, "r", encoding="utf-8") as f:
        vocab_tiny = json.load(f)
    
    vocab_new = {k: v["word"] for k, v in vocab_tiny["_default"].items()}
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab_new, f, ensure_ascii=False, indent=2)
    
    print(f"词表已保存到 {output_file}")
    return vocab_new


def freq_2_json(tinydb_file: str, output_file: str) -> List[Dict[str, int]]:
   
   
    with open(tinydb_file, "r", encoding="utf-8") as f:
        docs_tiny = json.load(f)
    
    docs_new = [doc_info["words"] for doc_info in docs_tiny["_default"].values()]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_new, f, ensure_ascii=False, indent=2)
    
    print(f"文章词频已保存到 {output_file}")
    return docs_new



if __name__ == "__main__":
    vocab =vocab_2_json(VOCAB_PATH, f"{DATA_PATH}/vocab.json")
    docs = freq_2_json(CONTENT_INFO, f"{DATA_PATH}freq.json")