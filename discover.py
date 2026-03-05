#!/usr/bin/env python3
"""
向量数据库信息探测工具

自动连接 Milvus / Qdrant，采集迁移所需的全部元信息。
用法:
  python discover.py --milvus-host localhost --milvus-port 19530
  python discover.py --qdrant-host localhost --qdrant-port 6333
  python discover.py --all                # 同时探测两个（使用默认地址）
"""

import argparse
import json
import sys


# ═══════════════════════════════════════════════════════
#  Milvus 探测
# ═══════════════════════════════════════════════════════
def discover_milvus(host: str, port: str):
    try:
        from pymilvus import connections, utility, Collection
    except ImportError:
        print("[Milvus] pymilvus 未安装，请执行: pip install pymilvus")
        return

    print("=" * 64)
    print("  Milvus 信息探测")
    print("=" * 64)

    try:
        connections.connect("default", host=host, port=port)
    except Exception as e:
        print(f"[Milvus] 连接失败 ({host}:{port}): {e}")
        return

    print(f"  主机地址:  {host}")
    print(f"  端口:      {port}")

    collection_names = utility.list_collections()
    if not collection_names:
        print("\n  (未发现任何 Collection)")
        connections.disconnect("default")
        return

    print(f"\n  发现 {len(collection_names)} 个 Collection:\n")

    for coll_name in sorted(collection_names):
        collection = Collection(coll_name)
        schema = collection.schema

        # 基本信息
        print(f"  ┌─ Collection: {coll_name}")
        print(f"  │  描述: {schema.description or '(无)'}")

        # 字段信息
        pk_field = None
        vector_fields = []
        scalar_fields = []

        for field in schema.fields:
            field_info = f"{field.name} ({field.dtype.name})"
            if field.is_primary:
                pk_field = field.name
                field_info += " [主键]"
            if field.dtype.name in ("FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR", "BINARY_VECTOR", "SPARSE_FLOAT_VECTOR"):
                dim = field.params.get("dim", "?")
                field_info += f" [维度={dim}]"
                vector_fields.append({"name": field.name, "dim": dim, "dtype": field.dtype.name})
            else:
                if not field.is_primary:
                    scalar_fields.append(field.name)

        print(f"  │  主键字段: {pk_field}")
        for vf in vector_fields:
            print(f"  │  向量字段: {vf['name']}  维度={vf['dim']}  类型={vf['dtype']}")
        print(f"  │  标量字段: {', '.join(scalar_fields) if scalar_fields else '(无)'}")

        # 索引和度量类型
        for vf in vector_fields:
            try:
                indexes = collection.indexes
                for idx in indexes:
                    if idx.field_name == vf["name"]:
                        params = idx.params
                        metric = params.get("metric_type", "未知")
                        index_type = params.get("index_type", "未知")
                        print(f"  │  索引类型: {index_type}")
                        print(f"  │  度量类型: {metric}")
                        extra = {k: v for k, v in params.items() if k not in ("metric_type", "index_type")}
                        if extra:
                            print(f"  │  索引参数: {extra}")
            except Exception:
                print(f"  │  索引信息: (无法获取，Collection 可能未建索引)")

        # 数据量
        try:
            num = collection.num_entities
            print(f"  │  数据量:   {num:,} 条")
        except Exception:
            print(f"  │  数据量:   (需要 flush 后才能获取准确值)")

        print(f"  └{'─' * 50}")
        print()

    connections.disconnect("default")


# ═══════════════════════════════════════════════════════
#  Qdrant 探测
# ═══════════════════════════════════════════════════════
def discover_qdrant(host: str, port: int):
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("[Qdrant] qdrant-client 未安装，请执行: pip install qdrant-client")
        return

    print("=" * 64)
    print("  Qdrant 信息探测")
    print("=" * 64)

    try:
        client = QdrantClient(host=host, port=port, timeout=10, check_compatibility=False)
        # 测试连通性
        client.get_collections()
    except Exception as e:
        print(f"[Qdrant] 连接失败 ({host}:{port}): {e}")
        return

    print(f"  主机地址:  {host}")
    print(f"  端口:      {port}")

    collections = client.get_collections().collections
    if not collections:
        print("\n  (未发现任何 Collection)")
        return

    print(f"\n  发现 {len(collections)} 个 Collection:\n")

    # 距离类型映射
    for coll_desc in sorted(collections, key=lambda c: c.name):
        coll_name = coll_desc.name
        info = client.get_collection(coll_name)

        print(f"  ┌─ Collection: {coll_name}")

        # 向量配置
        vectors_config = info.config.params.vectors
        if hasattr(vectors_config, "size"):
            # 单向量配置
            print(f"  │  向量维度:  {vectors_config.size}")
            print(f"  │  度量类型:  {vectors_config.distance.name}")
        elif isinstance(vectors_config, dict):
            # 命名向量配置
            for vec_name, vec_params in vectors_config.items():
                print(f"  │  向量字段: {vec_name}  维度={vec_params.size}  度量={vec_params.distance.name}")
        else:
            print(f"  │  向量配置: {vectors_config}")

        # 数据量
        print(f"  │  数据量:   {info.points_count:,} 条")
        print(f"  │  索引状态: {info.status.name}")

        # Payload 字段（通过采样推断）
        try:
            sample_points, _ = client.scroll(
                collection_name=coll_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if sample_points and sample_points[0].payload:
                payload_keys = list(sample_points[0].payload.keys())
                payload_types = []
                for k in payload_keys:
                    v = sample_points[0].payload[k]
                    payload_types.append(f"{k} ({type(v).__name__})")
                print(f"  │  Payload 字段: {', '.join(payload_types)}")
            else:
                print(f"  │  Payload 字段: (无)")
        except Exception:
            print(f"  │  Payload 字段: (无法采样)")

        # 索引配置
        hnsw = info.config.hnsw_config
        if hnsw:
            print(f"  │  HNSW 参数: m={hnsw.m}, ef_construct={hnsw.ef_construct}")

        print(f"  └{'─' * 50}")
        print()


# ═══════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="向量数据库信息探测工具 — 自动采集 Milvus / Qdrant 迁移所需元信息"
    )
    parser.add_argument("--milvus-host", default="localhost", help="Milvus 地址 (默认 localhost)")
    parser.add_argument("--milvus-port", default="19530", help="Milvus 端口 (默认 19530)")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant 地址 (默认 localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant 端口 (默认 6333)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--milvus", action="store_true", help="只探测 Milvus")
    group.add_argument("--qdrant", action="store_true", help="只探测 Qdrant")
    group.add_argument("--all", action="store_true", help="同时探测 Milvus 和 Qdrant")

    args = parser.parse_args()

    if args.milvus or args.all:
        discover_milvus(args.milvus_host, args.milvus_port)

    if args.qdrant or args.all:
        discover_qdrant(args.qdrant_host, args.qdrant_port)

    print("探测完成。请将以上信息填入 README.md 的信息收集表格中。")


if __name__ == "__main__":
    main()
