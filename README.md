# 向量数据库迁移工具：Milvus / Qdrant → OpenSearch

将向量数据从 Milvus 或 Qdrant 导出，并导入到 OpenSearch k-NN 索引中。

---

## 目录

- [架构概览](#架构概览)
- [前置条件](#前置条件)
- [信息收集](#信息收集)
- [环境部署](#环境部署)
- [配置修改](#配置修改)
- [执行迁移](#执行迁移)
- [结果验证](#结果验证)
- [常见问题](#常见问题)

---

## 架构概览

```
┌──────────┐    导出 JSONL     ┌──────────────┐    批量写入     ┌────────────┐
│  Milvus  │ ───────────────→ │              │ ─────────────→ │            │
└──────────┘                  │  本地磁盘     │               │ OpenSearch │
┌──────────┐    导出 JSONL     │  exports/    │    批量写入     │  (k-NN)   │
│  Qdrant  │ ───────────────→ │              │ ─────────────→ │            │
└──────────┘                  └──────────────┘               └────────────┘
```

**流程说明：**

1. 从 Milvus 按主键范围分批查询，导出为 JSONL 文件
2. 从 Qdrant 通过 scroll API 分批滚动导出为 JSONL 文件
3. 读取 JSONL 文件，通过 OpenSearch Bulk API 批量写入 k-NN 索引
4. 在 OpenSearch 上执行向量搜索验证数据正确性

---

## 前置条件

### 软件要求

| 组件 | 版本要求 |
|------|---------|
| Python | >= 3.8 |
| Docker & Docker Compose | Docker >= 20.10 |
| Milvus | v2.4.x（项目使用 v2.4.17） |
| Qdrant | v1.12.x（项目使用 v1.12.6） |
| OpenSearch | v2.x（项目使用 v2.18.0，需启用 k-NN 插件） |

### Python 依赖安装

```bash
pip install pymilvus qdrant-client opensearch-py numpy
```

---

## 信息收集

### 自动探测（推荐）

项目提供了 `discover.py` 探测脚本，可自动连接 Milvus / Qdrant 并采集迁移所需的全部元信息（Collection 名称、字段、维度、度量类型、数据量等）。

```bash
# 只探测 Milvus
python discover.py --milvus --milvus-host <地址> --milvus-port <端口>

# 只探测 Qdrant
python discover.py --qdrant --qdrant-host <地址> --qdrant-port <端口>

# 同时探测两个（默认 localhost）
python discover.py --all
```

**输出示例：**

```
================================================================
  Milvus 信息探测
================================================================
  主机地址:  localhost
  端口:      19530

  发现 2 个 Collection:

  ┌─ Collection: document_vectors
  │  描述: Document vector collection for semantic search
  │  主键字段: doc_id
  │  向量字段: vector  维度=256  类型=FLOAT_VECTOR
  │  标量字段: doc_type (INT64)
  │  索引类型: HNSW
  │  度量类型: COSINE
  │  索引参数: {'params': {'M': 16, 'efConstruction': 200}}
  │  数据量:   2,000 条
  └──────────────────────────────────────────────────

  ┌─ Collection: product_embeddings
  │  描述: Product embedding collection
  │  主键字段: product_id
  │  向量字段: embedding  维度=128  类型=FLOAT_VECTOR
  │  标量字段: category (INT64), tag (VARCHAR)
  │  索引类型: IVF_FLAT
  │  度量类型: L2
  │  索引参数: {'params': {'nlist': 128}}
  │  数据量:   5,000 条
  └──────────────────────────────────────────────────

================================================================
  Qdrant 信息探测
================================================================
  主机地址:  localhost
  端口:      6333

  发现 2 个 Collection:

  ┌─ Collection: image_features
  │  向量字段: (默认)  维度=512  度量=COSINE
  │  数据量:   3,000 条
  │  索引状态: GREEN
  │  Payload 字段: label (str), width (int), height (int)
  │  HNSW 参数: m=16, ef_construct=100
  └──────────────────────────────────────────────────

  ┌─ Collection: user_profiles
  │  向量字段: (默认)  维度=128  度量=EUCLID
  │  数据量:   5,000 条
  │  索引状态: GREEN
  │  Payload 字段: category (int), tag (str), score (float)
  │  HNSW 参数: m=16, ef_construct=100
  └──────────────────────────────────────────────────
```

> 请将探测结果对照下方表格确认，确保无遗漏。

### 手动收集

如果无法运行探测脚本，请手动收集以下信息。

### 源端 — Milvus

| 信息项 | 您的值 | 说明 |
|--------|--------|------|
| 主机地址 | __________ | Milvus gRPC 地址 |
| 端口 | __________ | 默认 `19530` |
| Collection 名称 | __________ | 要导出的集合 |
| 主键字段名 | __________ | 例如 `id` |
| 向量字段名 | __________ | 例如 `embedding` |
| 向量维度 | __________ | 例如 `128` |
| 距离度量类型 | __________ | `L2` / `IP` / `COSINE` |
| 标量字段列表 | __________ | 需一并导出的字段，逗号分隔 |
| 预估数据量（条） | __________ | 用于评估磁盘和耗时 |

### 源端 — Qdrant

| 信息项 | 您的值 | 说明 |
|--------|--------|------|
| 主机地址 | __________ | Qdrant REST API 地址 |
| 端口 | __________ | 默认 `6333` |
| Collection 名称 | __________ | 要导出的集合 |
| 向量维度 | __________ | 例如 `128` |
| 距离度量类型 | __________ | `Euclid` / `Cosine` / `Dot` |
| Payload 字段列表 | __________ | 需导出的 payload 字段，逗号分隔 |
| 预估数据量（条） | __________ | 用于评估磁盘和耗时 |

### 目标端 — OpenSearch

| 信息项 | 您的值 | 说明 |
|--------|--------|------|
| 主机地址 | __________ | OpenSearch 集群地址 |
| 端口 | __________ | 默认 `9200` |
| 是否启用 SSL | __________ | `是` / `否` |
| 用户名 | __________ | 如启用安全插件 |
| 密码 | __________ | 如启用安全插件 |
| 目标 Index 名称 | __________ | 例如 `vectors_from_milvus` |
| k-NN 引擎 | __________ | `lucene`（推荐）/ `nmslib` / `faiss` |
| HNSW ef_construction | __________ | 默认 `256` |
| HNSW m | __________ | 默认 `16` |
| 分片数 | __________ | 按集群规模决定，默认 `1` |
| 副本数 | __________ | 生产环境建议 >= 1，默认 `0` |

### 距离度量类型映射

迁移时需确保源端和目标端的度量类型一致，对应关系如下：

| Milvus | Qdrant | OpenSearch `space_type` |
|--------|--------|------------------------|
| `L2` | `Euclid` | `l2` |
| `IP` | `Dot` | `innerproduct` |
| `COSINE` | `Cosine` | `cosinesimil` |

---

## 环境部署

### 方式一：使用本项目 Docker Compose（测试环境）

本项目提供了包含 Milvus、Qdrant、OpenSearch 的 `docker-compose.yml`，适用于功能验证和测试。

```bash
# 启动所有服务
cd /home/ubuntu/vector
docker-compose up -d

# 查看服务状态
docker-compose ps

# 等待所有服务 healthy（约 1-2 分钟）
# 验证各服务连通性
curl http://localhost:9200/_cluster/health   # OpenSearch
curl http://localhost:6333/collections        # Qdrant
# Milvus gRPC 端口 19530 可通过 Python 脚本验证
```

### 方式二：连接已有服务（生产环境）

如果源端和目标端服务已经部署，只需修改脚本中的连接参数（见[配置修改](#配置修改)）。

---

## 配置修改

编辑 `export_import_opensearch.py` 文件头部的配置区域：

```python
# ── Config ──────────────────────────────────────────────
DIM = 128            # 向量维度，必须与源端一致
NUM_VECTORS = 10000  # 预期数据量（影响导出进度显示）
BATCH_SIZE = 1000    # 批量大小，可根据内存调整
TOP_K = 5            # 验证搜索时返回的结果数
NQ = 3               # 验证搜索的查询数量
```

### 连接地址修改

根据实际环境修改脚本中各数据库的连接参数：

**Milvus**（`milvus_insert_and_export` 函数内）：

```python
connections.connect("default", host="<MILVUS_HOST>", port="<MILVUS_PORT>")
```

**Qdrant**（`qdrant_insert_and_export` 函数内）：

```python
client = QdrantClient(host="<QDRANT_HOST>", port=<QDRANT_PORT>)
```

**OpenSearch**（`import_to_opensearch` 函数内）：

```python
client = OpenSearch(
    hosts=[{"host": "<OPENSEARCH_HOST>", "port": <OPENSEARCH_PORT>}],
    use_ssl=False,          # 如启用 SSL 改为 True
    # http_auth=("user", "password"),  # 如需认证，取消注释
)
```

---

## 执行迁移

### 完整流程（导出 + 导入 + 验证）

```bash
cd /home/ubuntu/vector
python export_import_opensearch.py
```

脚本会依次执行：

1. **Milvus 导出** — 按 ID 范围分批查询，写入 `exports/milvus_export.jsonl`
2. **Qdrant 导出** — 通过 scroll 分批滚动，写入 `exports/qdrant_export.jsonl`
3. **导入 OpenSearch** — 创建 k-NN 索引，Bulk 写入 Milvus 导出数据
4. **搜索验证** — 执行 k-NN 搜索和带过滤的搜索
5. **导入 OpenSearch** — 创建 k-NN 索引，Bulk 写入 Qdrant 导出数据
6. **搜索验证** — 同上

### 导出文件说明

执行后 `exports/` 目录下会生成以下文件：

```
exports/
├── milvus_export.jsonl   # Milvus 向量数据（每行一条 JSON）
├── milvus_meta.json      # Milvus 导出元信息（维度、度量类型、字段等）
├── qdrant_export.jsonl   # Qdrant 向量数据
└── qdrant_meta.json      # Qdrant 导出元信息
```

**JSONL 数据格式**（每行一条记录）：

```json
{"id": 0, "category": 5, "embedding": [0.123, 0.456, ...]}
```

**Meta 文件示例**（`milvus_meta.json`）：

```json
{
  "source": "milvus",
  "dim": 128,
  "metric_type": "L2",
  "count": 10000,
  "fields": ["id (INT64, PK)", "category (INT64)", "embedding (FLOAT_VECTOR)"],
  "index": {"type": "IVF_FLAT", "nlist": 128}
}
```

---

## 结果验证

脚本执行完毕后会打印汇总报告：

```
============================================================
  OPENSEARCH IMPORT SUMMARY
============================================================
Source                     Import (s)  Search (ms)  Filter (ms)
--------------------------------------------------------------
Milvus → OpenSearch             x.xxx        xx.xx        xx.xx
Qdrant → OpenSearch             x.xxx        xx.xx        xx.xx
============================================================
```

### 手动验证（可选）

```bash
# 检查索引是否创建成功
curl http://localhost:9200/_cat/indices?v

# 检查索引文档数量
curl http://localhost:9200/vectors_from_milvus/_count
curl http://localhost:9200/vectors_from_qdrant/_count

# 检查索引 mapping
curl http://localhost:9200/vectors_from_milvus/_mapping?pretty
```

---

## 常见问题

### Q1: OpenSearch 报错 `circuit_breaking_exception`

**原因：** JVM 堆内存不足，大批量写入时容易触发。

**解决：** 调整 `docker-compose.yml` 中 OpenSearch 的 JVM 参数：

```yaml
environment:
  - OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g   # 根据机器内存调大
```

或减小脚本中的 `BATCH_SIZE`。

### Q2: Milvus 导出报错 `collection not loaded`

**原因：** 查询前需要先将 Collection 加载到内存。

**解决：** 确保脚本中 `collection.load()` 在 `collection.query()` 之前被调用（当前脚本已包含此步骤）。

### Q3: 导出数据量与源端不一致

**排查步骤：**

1. 检查 `exports/*_meta.json` 中的 `count` 字段
2. 在源端确认实际数据量：
   - Milvus: `collection.num_entities`
   - Qdrant: `client.get_collection(name).points_count`
3. 如使用主键范围导出，确认范围覆盖了所有数据

### Q4: OpenSearch 搜索结果与源端差异较大

**可能原因：**

- 度量类型映射错误（如源端用 `Cosine`，目标端配成了 `l2`）
- HNSW 参数差异导致召回率不同，可增大 `ef_construction` 提升精度

### Q5: 磁盘空间估算

JSONL 中间文件大小参考：

| 向量维度 | 每万条约占空间 |
|---------|---------------|
| 128 | ~6 MB |
| 256 | ~12 MB |
| 512 | ~24 MB |
| 768 | ~35 MB |
| 1536 | ~70 MB |

请确保 `exports/` 所在磁盘有足够剩余空间。

### Q6: 如何只执行部分步骤？

当前脚本在 `__main__` 中串行执行所有步骤。如果只需执行部分操作，可以在 Python 中单独调用对应函数：

```python
from export_import_opensearch import *

# 只从 Milvus 导出
milvus_insert_and_export()

# 只导入到 OpenSearch（使用已有的 JSONL 文件）
from pathlib import Path
client, idx, t = import_to_opensearch(Path("exports/milvus_export.jsonl"), "my_index")

# 只验证搜索
search_opensearch(client, "my_index")
```

---

## 文件结构

```
vector/
├── README.md                      # 本操作手册
├── docker-compose.yml             # 测试环境一键部署（Milvus + Qdrant + OpenSearch）
├── discover.py                    # 源端信息自动探测脚本
├── export_import_opensearch.py    # 导出导入主脚本
├── vector_test.py                 # Milvus / Qdrant 性能对比测试（可选）
└── exports/                       # 导出数据目录
    ├── milvus_export.jsonl
    ├── milvus_meta.json
    ├── qdrant_export.jsonl
    └── qdrant_meta.json
```
