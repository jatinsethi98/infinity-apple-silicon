# What serving a RAG index costs: Mac mini vs AWS vs Pinecone

Back-of-the-envelope, dated 2026-09-03. Throughput comes from [BENCHMARKS.md](BENCHMARKS.md);
prices are public list prices on that date and will drift. The point is the shape of the curves,
not the cents.

## The short version

For a 1,000,000-vector, 768-dimensional index serving 2,000,000 queries a month:

| where it runs | monthly cost | what that buys |
|---|---:|---|
| Pinecone serverless, Standard plan | about $117 | pay per read unit, scales with index size and traffic |
| AWS m8g.xlarge (4 vCPU, 16 GiB Graviton4), on demand | about $135 | flat; roughly $91 on a 1-year reservation |
| Mac mini M4 (16 GB) you already own, electricity | about $0.54 | flat; the box is busy 0.02% of the month |
| the same Mac mini with its $599 price amortized over 3 years | about $17 | flat |

Pinecone is free on its Starter tier while the index is under 2 GB and traffic under 1M read
units a month, which covers a 100,000-vector side project. Above that its cost grows linearly with
both index size and query volume, because a query costs one read unit per gigabyte of namespace.
Self-hosted cost is flat and, on this hardware, nearly zero, because the engine's measured ceiling
(about 5,000 QPS at 768-d) is three orders of magnitude above these traffic levels.

## By scenario

| scenario | Pinecone index | RU per query | Pinecone / month | Mac mini electricity / month | Infinity RAM needed |
|---|---:|---:|---:|---:|---:|
| 100k × 768-d, 200k queries/mo, full monthly re-ingest | 0.36 GB | 0.36 | $0 (Starter) | $0.54 | 0.3 GB |
| 1M × 768-d, 2M queries/mo, 10% churn | 3.57 GB | 3.57 | $117 | $0.54 | 3.3 GB |
| 1M × 1536-d, 10M queries/mo, 10% churn | 6.64 GB | 6.64 | $1,068 | $0.54 | 6.4 GB |
| 3M × 768-d, 20M queries/mo, 10% churn | 10.72 GB | 10.72 | $3,437 | $0.54 | 10.0 GB |

Marginal cost of the next million operations on a 1M-vector index:

| | Pinecone | Mac mini electricity |
|---|---:|---:|
| 1M queries, 768-d | $57 | $0.0003 (200 s of CPU) |
| 1M queries, 1536-d | $106 | $0.0005 (311 s of CPU) |
| ingest 1M vectors, 768-d | $14 | $0.0003 (201 s of CPU) |
| ingest 1M vectors, 1536-d | $26 | $0.0005 (327 s of CPU) |

## Throughput ceilings

| | Pinecone serverless, documented limits | Infinity on the M4 mini, measured |
|---|---|---|
| ingest, 768-d | 50 MB/s per namespace, about 14,000 records/s; bulk import is asynchronous | 4,977 vectors/s HNSW build |
| ingest, 1536-d | about 7,500 records/s | 3,055 vectors/s |
| queries, 1M × 768-d | 100 QPS per namespace; 2,000 RU/s per index, about 560 QPS | 4,996 QPS in-process |
| queries, 1M × 1536-d | 100 QPS per namespace; about 300 QPS per index | 3,215 QPS in-process |

## Assumptions

- **Pinecone Standard:** $16 per million read units, $4 per million write units, $0.33 per
  GB-month, $50 monthly minimum. Starter: free to 2 GB, 1M RU and 2M WU per month. Builder: $20
  flat to 10 GB, 2M RU and 5M WU. One query costs 1 RU per GB of namespace, minimum 0.25; one
  upsert costs 1 WU per KB. Records are modelled as dims × 4 bytes plus 500 bytes of metadata.
- **Electricity:** 18.34 ¢/kWh, US residential average (EIA Electric Power Monthly, June 2026
  data). Commercial average 14.19 ¢/kWh.
- **Power:** Apple's published Mac mini M4 figures, 4 W idle and 65 W maximum; an all-core load is
  modelled at 35 W at the wall. Not measured directly. Even pegged at 65 W all month the box costs
  $8.70.
- **Hardware:** $599, the M4 mini's launch price, straight-line over 36 months. Apple's price in
  September 2026 is $799.
- **AWS:** us-east-1 Linux on-demand, m8g.xlarge at $0.180/h plus 50 GB gp3. A 4-vCPU Graviton
  will not match a 10-core M4, so treat it as a floor on AWS cost, not a throughput equivalent. A
  dedicated `mac2-m2.metal` host is $0.878/h, about $641 a month.
- **Not counted:** home internet and its lack of an SLA, backups, monitoring, and the engineer's
  time. Pinecone's per-query price includes all of that.
- **In-process QPS.** A server in front of the engine will lose some throughput to HTTP.

## Sources

- pinecone.io/pricing; docs.pinecone.io/guides/manage-cost/understanding-cost;
  docs.pinecone.io/reference/api/database-limits/rate-limits
- eia.gov/electricity/monthly, table 5.6.A
- support.apple.com/103253 (Mac mini power consumption and thermal output)
- instances.vantage.sh for EC2 on-demand prices; aws.amazon.com/ebs/pricing
