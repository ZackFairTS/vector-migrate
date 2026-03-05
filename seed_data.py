#!/usr/bin/env python3
"""
写入持久化测试数据到 Milvus 和 Qdrant，用于 discover.py 探测验证。
数据写入后不会删除，保留在数据库中。
"""

import numpy as np

DIM = 128
NUM_VECTORS = 5000
BATCH_SIZE = 1000

np.random.seed(42)
vectors = np.random.random((NUM_VECTORS, DIM)).astype(np.float32)
ids = list(range(NUM_VECTORS))
categories = np.random.randint(0, 10, NUM_VECTORS).tolist()
tags = [f"tag_{i % 20}" for i in range(NUM_VECTORS)]


def seed_milvus():
    from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

    print("=" * 60)
    print("  写入 Milvus 测试数据")
    print("=" * 60)

    connections.connect("default", host="localhost", port="19530")

    # Collection 1: 产品向量库
    name1 = "product_embeddings"
    if utility.has_collection(name1):
        utility.drop_collection(name1)

    schema1 = CollectionSchema([
        FieldSchema(name="product_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="category", dtype=DataType.INT64),
        FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ], description="Product embedding collection")
    coll1 = Collection(name1, schema1)

    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_VECTORS)
        coll1.insert([ids[i:end], categories[i:end], tags[i:end], vectors[i:end].tolist()])
    coll1.flush()

    coll1.create_index("embedding", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    })
    coll1.load()
    print(f"  [Milvus] {name1}: {coll1.num_entities} 条，L2, IVF_FLAT")

    # Collection 2: 文档向量库 (Cosine)
    name2 = "document_vectors"
    if utility.has_collection(name2):
        utility.drop_collection(name2)

    schema2 = CollectionSchema([
        FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="doc_type", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=256),
    ], description="Document vector collection for semantic search")
    coll2 = Collection(name2, schema2)

    vectors_256 = np.random.random((2000, 256)).astype(np.float32)
    doc_types = np.random.randint(0, 5, 2000).tolist()
    for i in range(0, 2000, BATCH_SIZE):
        end = min(i + BATCH_SIZE, 2000)
        coll2.insert([list(range(i, end)), doc_types[i:end], vectors_256[i:end].tolist()])
    coll2.flush()

    coll2.create_index("vector", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    })
    coll2.load()
    print(f"  [Milvus] {name2}: {coll2.num_entities} 条，COSINE, HNSW")

    connections.disconnect("default")
    print("  [Milvus] 数据写入完成，已保留在数据库中\n")


def seed_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    print("=" * 60)
    print("  写入 Qdrant 测试数据")
    print("=" * 60)

    client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

    # Collection 1: 用户向量库
    name1 = "user_profiles"
    if client.collection_exists(name1):
        client.delete_collection(name1)
    client.create_collection(
        collection_name=name1,
        vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
    )

    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_VECTORS)
        points = [
            PointStruct(
                id=ids[j],
                vector=vectors[j].tolist(),
                payload={"category": categories[j], "tag": tags[j], "score": round(float(np.random.random()), 4)},
            )
            for j in range(i, end)
        ]
        client.upsert(collection_name=name1, points=points)
    info1 = client.get_collection(name1)
    print(f"  [Qdrant] {name1}: {info1.points_count} 条，Euclid")

    # Collection 2: 图片向量库 (Cosine)
    name2 = "image_features"
    if client.collection_exists(name2):
        client.delete_collection(name2)
    client.create_collection(
        collection_name=name2,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    vectors_512 = np.random.random((3000, 512)).astype(np.float32)
    for i in range(0, 3000, BATCH_SIZE):
        end = min(i + BATCH_SIZE, 3000)
        points = [
            PointStruct(
                id=j,
                vector=vectors_512[j].tolist(),
                payload={"label": f"class_{j % 50}", "width": int(np.random.randint(100, 1920)), "height": int(np.random.randint(100, 1080))},
            )
            for j in range(i, end)
        ]
        client.upsert(collection_name=name2, points=points)
    info2 = client.get_collection(name2)
    print(f"  [Qdrant] {name2}: {info2.points_count} 条，Cosine")

    print("  [Qdrant] 数据写入完成，已保留在数据库中\n")


if __name__ == "__main__":
    seed_milvus()
    seed_qdrant()
    print("全部测试数据写入完成。请运行 python discover.py --all 查看探测结果。")
