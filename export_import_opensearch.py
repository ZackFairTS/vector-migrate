"""
Full pipeline:
  1. Insert test data into Milvus & Qdrant
  2. Export data from both to JSONL files
  3. Import JSONL into OpenSearch k-NN index
  4. Run vector search on OpenSearch and compare results
"""

import json
import time
import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────
DIM = 128
NUM_VECTORS = 10000
BATCH_SIZE = 1000
TOP_K = 5
NQ = 3

EXPORT_DIR = Path("/home/ubuntu/vector/exports")
EXPORT_DIR.mkdir(exist_ok=True)

np.random.seed(42)
vectors = np.random.random((NUM_VECTORS, DIM)).astype(np.float32)
ids = list(range(NUM_VECTORS))
categories = np.random.randint(0, 10, NUM_VECTORS).tolist()
query_vectors = np.random.random((NQ, DIM)).astype(np.float32)


# ═══════════════════════════════════════════════════════
#  STEP 1 & 2: Insert into Milvus, then export
# ═══════════════════════════════════════════════════════
def milvus_insert_and_export():
    from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

    print("=" * 60)
    print("  MILVUS: Insert → Export")
    print("=" * 60)

    connections.connect("default", host="localhost", port="19530")
    coll_name = "export_collection"
    if utility.has_collection(coll_name):
        utility.drop_collection(coll_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="category", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="export test")
    collection = Collection(coll_name, schema)

    # Insert
    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_VECTORS)
        collection.insert([ids[i:end], categories[i:end], vectors[i:end].tolist()])
    collection.flush()
    print(f"[Milvus] Inserted {collection.num_entities} vectors")

    # Build index & load for query
    collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    collection.load()

    # Export: query all data in batches
    export_file = EXPORT_DIR / "milvus_export.jsonl"
    count = 0
    with open(export_file, "w") as f:
        # Use query with expr to fetch all
        batch = 0
        while batch * BATCH_SIZE < NUM_VECTORS:
            lo = batch * BATCH_SIZE
            hi = min(lo + BATCH_SIZE, NUM_VECTORS)
            expr = f"id >= {lo} and id < {hi}"
            results = collection.query(
                expr=expr,
                output_fields=["id", "category", "embedding"],
            )
            for row in results:
                record = {
                    "id": row["id"],
                    "category": row["category"],
                    "embedding": row["embedding"],
                }
                f.write(json.dumps(record) + "\n")
                count += 1
            batch += 1

    print(f"[Milvus] Exported {count} records → {export_file}")

    # Save meta
    meta = {
        "source": "milvus",
        "dim": DIM,
        "metric_type": "L2",
        "count": count,
        "fields": ["id (INT64, PK)", "category (INT64)", "embedding (FLOAT_VECTOR)"],
        "index": {"type": "IVF_FLAT", "nlist": 128},
    }
    meta_file = EXPORT_DIR / "milvus_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Milvus] Meta → {meta_file}")

    collection.release()
    utility.drop_collection(coll_name)
    connections.disconnect("default")


# ═══════════════════════════════════════════════════════
#  STEP 1 & 2: Insert into Qdrant, then export
# ═══════════════════════════════════════════════════════
def qdrant_insert_and_export():
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    print("\n" + "=" * 60)
    print("  QDRANT: Insert → Export")
    print("=" * 60)

    client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
    coll_name = "export_collection"

    if client.collection_exists(coll_name):
        client.delete_collection(coll_name)
    client.create_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
    )

    # Insert
    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_VECTORS)
        points = [
            PointStruct(id=ids[j], vector=vectors[j].tolist(), payload={"category": categories[j]})
            for j in range(i, end)
        ]
        client.upsert(collection_name=coll_name, points=points)
    info = client.get_collection(coll_name)
    print(f"[Qdrant] Inserted {info.points_count} vectors")

    # Export: scroll all points
    export_file = EXPORT_DIR / "qdrant_export.jsonl"
    count = 0
    with open(export_file, "w") as f:
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=coll_name,
                limit=BATCH_SIZE,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            if not result:
                break
            for pt in result:
                record = {
                    "id": pt.id,
                    "category": pt.payload.get("category"),
                    "embedding": pt.vector,
                }
                f.write(json.dumps(record) + "\n")
                count += 1
            offset = next_offset
            if offset is None:
                break

    print(f"[Qdrant] Exported {count} records → {export_file}")

    meta = {
        "source": "qdrant",
        "dim": DIM,
        "metric_type": "Euclid",
        "count": count,
        "fields": ["id (int)", "category (int, payload)", "embedding (float32 vector)"],
    }
    meta_file = EXPORT_DIR / "qdrant_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Qdrant] Meta → {meta_file}")

    client.delete_collection(coll_name)


# ═══════════════════════════════════════════════════════
#  STEP 3: Import JSONL into OpenSearch
# ═══════════════════════════════════════════════════════
def import_to_opensearch(jsonl_path: Path, index_name: str):
    from opensearchpy import OpenSearch, helpers

    print(f"\n{'=' * 60}")
    print(f"  OPENSEARCH: Import ← {jsonl_path.name} → index '{index_name}'")
    print("=" * 60)

    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False,
    )

    # Delete index if exists
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    # Create index with k-NN mapping
    body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "integer"},
                "category": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 256,
                            "m": 16,
                        },
                    },
                },
            }
        },
    }
    client.indices.create(index=index_name, body=body)
    print(f"[OpenSearch] Created index '{index_name}' (k-NN, HNSW, L2)")

    # Bulk insert from JSONL
    def gen_actions():
        with open(jsonl_path) as f:
            for line in f:
                doc = json.loads(line)
                yield {
                    "_index": index_name,
                    "_id": doc["id"],
                    "_source": {
                        "id": doc["id"],
                        "category": doc["category"],
                        "embedding": doc["embedding"],
                    },
                }

    t0 = time.perf_counter()
    success, errors = helpers.bulk(client, gen_actions(), chunk_size=BATCH_SIZE)
    insert_time = time.perf_counter() - t0
    print(f"[OpenSearch] Bulk inserted {success} docs in {insert_time:.3f}s (errors: {len(errors) if isinstance(errors, list) else errors})")

    # Refresh to make data searchable
    client.indices.refresh(index=index_name)
    count = client.count(index=index_name)["count"]
    print(f"[OpenSearch] Index doc count: {count}")

    return client, index_name, insert_time


# ═══════════════════════════════════════════════════════
#  STEP 4: Vector search on OpenSearch
# ═══════════════════════════════════════════════════════
def search_opensearch(client, index_name: str):
    print(f"\n{'=' * 60}")
    print(f"  OPENSEARCH: Vector Search on '{index_name}'")
    print("=" * 60)

    # Basic k-NN search
    t0 = time.perf_counter()
    for qi in range(NQ):
        body = {
            "size": TOP_K,
            "_source": ["id", "category"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vectors[qi].tolist(),
                        "k": TOP_K,
                    }
                }
            },
        }
        resp = client.search(index=index_name, body=body)
        if qi == 0:
            first_resp = resp
    search_time = time.perf_counter() - t0

    print(f"[OpenSearch] Search {NQ} queries x top-{TOP_K} in {search_time * 1000:.2f}ms")
    for qi in range(NQ):
        body = {
            "size": TOP_K,
            "_source": ["id", "category"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vectors[qi].tolist(),
                        "k": TOP_K,
                    }
                }
            },
        }
        resp = client.search(index=index_name, body=body)
        print(f"  Query {qi}:")
        for hit in resp["hits"]["hits"]:
            print(f"    id={hit['_source']['id']:>5d}  score={hit['_score']:.6f}  category={hit['_source']['category']}")

    # Filtered search (category == 3) using post_filter
    t0 = time.perf_counter()
    body = {
        "size": TOP_K,
        "_source": ["id", "category"],
        "query": {
            "bool": {
                "must": {
                    "knn": {
                        "embedding": {
                            "vector": query_vectors[0].tolist(),
                            "k": 50,
                        }
                    }
                },
                "filter": {
                    "term": {"category": 3}
                },
            }
        },
    }
    resp = client.search(index=index_name, body=body)
    filter_time = time.perf_counter() - t0
    print(f"\n[OpenSearch] Filtered search (category==3) in {filter_time * 1000:.2f}ms")
    for hit in resp["hits"]["hits"]:
        print(f"    id={hit['_source']['id']:>5d}  score={hit['_score']:.6f}  category={hit['_source']['category']}")

    return search_time, filter_time


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1-2: Insert & Export
    milvus_insert_and_export()
    qdrant_insert_and_export()

    # Show exported files
    print(f"\n{'=' * 60}")
    print("  EXPORTED FILES")
    print("=" * 60)
    for p in sorted(EXPORT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<30s} {size_kb:>10.1f} KB")

    # Step 3-4: Import & Search (from Milvus export)
    os_client, idx, milvus_insert_t = import_to_opensearch(
        EXPORT_DIR / "milvus_export.jsonl", "vectors_from_milvus"
    )
    m_search_t, m_filter_t = search_opensearch(os_client, idx)

    # Step 3-4: Import & Search (from Qdrant export)
    os_client2, idx2, qdrant_insert_t = import_to_opensearch(
        EXPORT_DIR / "qdrant_export.jsonl", "vectors_from_qdrant"
    )
    q_search_t, q_filter_t = search_opensearch(os_client2, idx2)

    # Summary
    print(f"\n{'=' * 60}")
    print("  OPENSEARCH IMPORT SUMMARY")
    print("=" * 60)
    print(f"{'Source':<25} {'Import (s)':>12} {'Search (ms)':>12} {'Filter (ms)':>12}")
    print("-" * 62)
    print(f"{'Milvus → OpenSearch':<25} {milvus_insert_t:>12.3f} {m_search_t * 1000:>12.2f} {m_filter_t * 1000:>12.2f}")
    print(f"{'Qdrant → OpenSearch':<25} {qdrant_insert_t:>12.3f} {q_search_t * 1000:>12.2f} {q_filter_t * 1000:>12.2f}")
    print("=" * 60)

    # Cleanup OpenSearch indices
    os_client.indices.delete(index="vectors_from_milvus")
    os_client2.indices.delete(index="vectors_from_qdrant")
    print("[OpenSearch] Cleaned up indices.")
