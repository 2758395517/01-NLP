# database.py - 改进版
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class MedicalVectorDatabase:
    def __init__(self, model_name="moka-ai/m3e-base"):
        self.embed_model = SentenceTransformer(model_name)
        self.dimension = self.embed_model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.metadata = []

    def build_from_chunks(self, chunks_file="medical_chunks.json", batch_size=256):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"总共 {len(chunks)} 个chunks")

        # 提取文本
        texts = [chunk["text"] for chunk in chunks]

        # 批量生成嵌入
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="生成向量"):
            batch_texts = texts[i:i + batch_size]
            batch_embeds = self.embed_model.encode(
                batch_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeds)

        embeddings = np.vstack(embeddings).astype('float32')
        print(f"向量维度: {embeddings.shape}")

        # 创建索引 - 使用更高效的IndexFlatIP（内积相似度）
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks
        self.metadata = [chunk.get("metadata", {}) for chunk in chunks]

        print(f"向量数据库构建完成，共 {len(chunks)} 个文档")

    def search(self, query, top_k=5, threshold=0.6):
        if self.index is None or len(self.chunks) == 0:
            return []

        # 查询向量化
        query_embedding = self.embed_model.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')

        # 搜索
        scores, indices = self.index.search(query_embedding, min(top_k * 3, len(self.chunks)))

        results = []
        seen_ids = set()

        # 过滤和整理结果
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue

            # 去重：防止相似内容重复出现
            chunk_id = self.chunks[idx].get("id", idx)
            if chunk_id in seen_ids:
                continue

            # 阈值过滤
            if score < threshold:
                continue

            # 内容相关性额外检查
            chunk_text = self.chunks[idx].get("text", "").lower()
            query_terms = query.lower().split()
            text_match_score = sum(1 for term in query_terms if term in chunk_text) / len(query_terms)

            # 综合分数
            combined_score = 0.7 * score + 0.3 * text_match_score

            if combined_score < threshold:
                continue

            result = {
                "id": int(chunk_id),
                "chunk": self.chunks[idx],
                "score": float(combined_score),
                "text": self.chunks[idx].get("text", ""),
                "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                "question": self.metadata[idx].get("question", "") if idx < len(self.metadata) else "",
                "answer": self.metadata[idx].get("answer", "") if idx < len(self.metadata) else "",
                "department": self.metadata[idx].get("department", "") if idx < len(self.metadata) else ""
            }

            results.append(result)
            seen_ids.add(chunk_id)

            if len(results) >= top_k:
                break

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path="medical_vector_db"):
        if self.index is not None:
            faiss.write_index(self.index, f"{path}.index")

        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f)
        print(f"向量数据库已保存到 {path}")

    def load(self, path="medical_vector_db"):
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        else:
            return False

        with open(f"{path}.meta.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]

        print(f"向量数据库加载完成，共 {len(self.chunks)} 个文档")
        return True


def build_vector_database():
    if not os.path.exists("medical_chunks.json"):
        print("错误: 未找到 medical_chunks.json 文件")
        print("请先运行 data.py 生成数据")
        return None

    vector_db = MedicalVectorDatabase(model_name="moka-ai/m3e-base")
    vector_db.build_from_chunks("medical_chunks.json")
    vector_db.save()
    return vector_db


def load_vector_database():

    if os.path.exists("medical_vector_db.index") and os.path.exists("medical_vector_db.meta.pkl"):
        vector_db = MedicalVectorDatabase(model_name="moka-ai/m3e-base")
        if vector_db.load():
            return vector_db
    return None


def get_vector_database():
    vector_db = load_vector_database()
    if vector_db is None:
        print("未找到现有向量数据库，正在构建...")
        vector_db = build_vector_database()
    return vector_db


if __name__ == "__main__":
    # 测试向量数据库
    print("=" * 50)
    print("测试向量数据库")
    print("=" * 50)

    vector_db = get_vector_database()

    if vector_db:
        test_queries = [
            "高血压应该吃什么药？",
            "感冒了怎么办？",
            "胃痛怎么缓解？",
            "糖尿病患者可以吃什么水果？",
            "心脏病有哪些症状？"
        ]

        for query in test_queries:
            print(f"\n查询: {query}")
            results = vector_db.search(query, top_k=3, threshold=0.5)

            if results:
                for i, result in enumerate(results):
                    print(f"  结果 {i + 1} (相似度: {result['score']:.3f}):")
                    print(f"    科室: {result['metadata'].get('department', '未知')}")
                    print(f"    问题: {result.get('question', '')[:60]}...")
                    print(f"    答案: {result.get('answer', '')[:80]}...")
            else:
                print("  未找到相关结果")
    else:
        print("向量数据库初始化失败")