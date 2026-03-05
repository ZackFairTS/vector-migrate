"""
Vector Database Benchmark: Milvus vs Qdrant
- Insert random vectors
- Perform similarity search (L2 & Cosine)
- Compare results and latency
"""

import time
import numpy as np
from pymilvus import (
    connections, utility, Collection,
    CollectionSchema, FieldSchema, DataType,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
)

# ── Config ──────────────────────────────────────────────
DIM = 128           # vector dimension
NUM_VECTORS = 10000 # total vectors to insert
BATCH_SIZE = 1000   # insert batch size
TOP_K = 5           # search top-k
NQ = 3              # number of query vectors

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

np.random.seed(42)

# ── Generate data ───────────────────────────────────────
print("=" * 60)
print(f"Generating {NUM_VECTORS} random vectors (dim={DIM}) ...")
vectors = np.random.random((NUM_VECTORS, DIM)).astype(np.float32)
ids = list(range(NUM_VECTORS))
# Add a category field to demonstrate filtering
categories = np.random.randint(0, 10, NUM_VECTORS).tolist()
query_vectors = np.random.random((NQ, DIM)).astype(np.float32)
print("Data generation done.\n")


# ═══════════════════════════════════════════════════════
#  MILVUS TEST
# ═══════════════════════════════════════════════════════
def test_milvus():
    print("=" * 60)
    print("  MILVUS TEST")
    print("=" * 60)

    # Connect
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"[Milvus] Connected to {MILVUS_HOST}:{MILVUS_PORT}")

    collection_name = "test_collection"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"[Milvus] Dropped existing collection '{collection_name}'")

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="category", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="vector test collection")
    collection = Collection(collection_name, schema)
    print(f"[Milvus] Created collection '{collection_name}'")

    # Insert
    t0 = time.perf_counter()
    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_VECTORS)
        collection.insert([
            ids[i:end],
            categories[i:end],
            vectors[i:end].tolist(),
        ])
    collection.flush()
    insert_time = time.perf_counter() - t0
    print(f"[Milvus] Inserted {NUM_VECTORS} vectors in {insert_time:.3f}s")
    print(f"[Milvus] Collection count: {collection.num_entities}")

    # Create index
    t0 = time.perf_counter()
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index_params)
    index_time = time.perf_counter() - t0
    print(f"[Milvus] Index built (IVF_FLAT, nlist=128) in {index_time:.3f}s")

    # Load into memory
    collection.load()

    # Search
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    t0 = time.perf_counter()
    results = collection.search(
        data=query_vectors.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["category"],
    )
    search_time = time.perf_counter() - t0
    print(f"\n[Milvus] Search {NQ} queries x top-{TOP_K} in {search_time*1000:.2f}ms")

    for qi, hits in enumerate(results):
        print(f"  Query {qi}:")
        for hit in hits:
            print(f"    id={hit.id:>5d}  distance={hit.distance:.6f}  category={hit.entity.get('category')}")

    # Search with filter
    t0 = time.perf_counter()
    results_filtered = collection.search(
        data=query_vectors[:1].tolist(),
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        expr="category == 3",
        output_fields=["category"],
    )
    filter_time = time.perf_counter() - t0
    print(f"\n[Milvus] Filtered search (category==3) in {filter_time*1000:.2f}ms")
    for hit in results_filtered[0]:
        print(f"    id={hit.id:>5d}  distance={hit.distance:.6f}  category={hit.entity.get('category')}")

    # Cleanup
    collection.release()
    utility.drop_collection(collection_name)
    connections.disconnect("default")
    print(f"\n[Milvus] Cleanup done.")
    return insert_time, search_time, filter_time


# ═══════════════════════════════════════════════════════
#  QDRANT TEST
# ═══════════════════════════════════════════════════════
def test_qdrant():
    print("\n" + "=" * 60)
    print("  QDRANT TEST")
    print("=" * 60)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"[Qdrant] Connected to {QDRANT_HOST}:{QDRANT_PORT}")

    collection_name = "test_collection"
    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
    )
    print(f"[Qdrant] Created collection '{collection_name}' (EUCLID)")

    # Insert
    t0 = time.perf_counter()
    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_VECTORS)
        points = [
            PointStruct(
                id=ids[j],
                vector=vectors[j].tolist(),
                payload={"category": categories[j]},
            )
            for j in range(i, end)
        ]
        client.upsert(collection_name=collection_name, points=points)
    insert_time = time.perf_counter() - t0
    print(f"[Qdrant] Inserted {NUM_VECTORS} vectors in {insert_time:.3f}s")

    info = client.get_collection(collection_name)
    print(f"[Qdrant] Collection point count: {info.points_count}")

    # Search
    t0 = time.perf_counter()
    all_results = []
    for qi in range(NQ):
        res = client.query_points(
            collection_name=collection_name,
            query=query_vectors[qi].tolist(),
            limit=TOP_K,
            with_payload=True,
        )
        all_results.append(res)
    search_time = time.perf_counter() - t0
    print(f"\n[Qdrant] Search {NQ} queries x top-{TOP_K} in {search_time*1000:.2f}ms")

    for qi, res in enumerate(all_results):
        print(f"  Query {qi}:")
        for pt in res.points:
            print(f"    id={pt.id:>5d}  score={pt.score:.6f}  category={pt.payload.get('category')}")

    # Search with filter
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    t0 = time.perf_counter()
    filtered_res = client.query_points(
        collection_name=collection_name,
        query=query_vectors[0].tolist(),
        query_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=3))]
        ),
        limit=TOP_K,
        with_payload=True,
    )
    filter_time = time.perf_counter() - t0
    print(f"\n[Qdrant] Filtered search (category==3) in {filter_time*1000:.2f}ms")
    for pt in filtered_res.points:
        print(f"    id={pt.id:>5d}  score={pt.score:.6f}  category={pt.payload.get('category')}")

    # Cleanup
    client.delete_collection(collection_name)
    print(f"\n[Qdrant] Cleanup done.")
    return insert_time, search_time, filter_time


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    m_insert, m_search, m_filter = test_milvus()
    q_insert, q_search, q_filter = test_qdrant()

    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Milvus':>12} {'Qdrant':>12}")
    print("-" * 50)
    print(f"{'Insert (s)':<25} {m_insert:>12.3f} {q_insert:>12.3f}")
    print(f"{'Search (ms)':<25} {m_search*1000:>12.2f} {q_search*1000:>12.2f}")
    print(f"{'Filtered Search (ms)':<25} {m_filter*1000:>12.2f} {q_filter*1000:>12.2f}")
    print(f"{'Vectors':<25} {NUM_VECTORS:>12d} {NUM_VECTORS:>12d}")
    print(f"{'Dimension':<25} {DIM:>12d} {DIM:>12d}")
    print("=" * 60)
