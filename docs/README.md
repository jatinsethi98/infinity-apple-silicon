# Documentation

Everything here is about running Infinity natively on an Apple Silicon Mac. The pages
are in the order most people need them.

## Start here

| Page | What it answers |
|---|---|
| [Getting started](getting-started.md) | How do I install it, start it, stop it, and fix a setup problem? |
| [Concepts](concepts.md) | What are embeddings, HNSW, recall and hybrid search, in plain language? |
| [Python guide](guides/python.md) | How do I create tables, insert, index and search from Python? |
| [HTTP guide](guides/http.md) | How do I do the same from any language with `curl`? |
| [RAG tutorial](guides/rag.md) | How do I build a retrieval-augmented app on top of it? |

## Running it

| Page | What it answers |
|---|---|
| [Operations](guides/operations.md) | Instances, ports, configuration, logs, data directories, snapshots, durability. |
| [Tuning](guides/tuning.md) | Which index parameters, thread counts and memory settings matter, with measured effects. |
| [Benchmarking](guides/benchmarking.md) | How to reproduce the published numbers, and how to measure a change fairly. |
| [Known issues](known-issues.md) | What is broken today, how to avoid it, and where each defect is tracked. |
| [FAQ](faq.md) | Short answers to the questions that come up most. |

## Reference

The API reference is upstream Infinity's, carried unchanged because the server speaks
the same protocol. Where this port differs (paths, ports, the absence of an embedded
mode) the guides above say so.

| Document | |
|---|---|
| [Python SDK reference](references/pysdk_api_reference.md) | Every SDK call with parameters and examples. |
| [HTTP API reference](references/http_api_reference.mdx) | Every endpoint with request and response bodies. |
| [Search syntax guide](guides/search_guide.md) | Full-text query syntax, dense, sparse and tensor search, fusion, filters. |
| [Configuration reference](references/configurations.mdx) | Every key in `infinity_conf.toml`. |

## Engineering record

The evidence behind every claim in the README, kept in the style of a lab notebook.
Read these when you want to check a number, understand a design decision, or pick up
an open thread.

| Document | |
|---|---|
| [apple_silicon/README.md](apple_silicon/README.md) | Building by hand, platform boundaries, SIMD and allocator notes, the HNSW convention difference from FAISS. |
| [apple_silicon/BENCHMARKS.md](apple_silicon/BENCHMARKS.md) | Published results, method, fairness rules, limitations, reproduction commands. |
| [apple_silicon/MACOS_VERIFICATION.md](apple_silicon/MACOS_VERIFICATION.md) | What is verified on macOS and what is not, with the command behind each claim. |
| [apple_silicon/EVALUATION.md](apple_silicon/EVALUATION.md) | The adversarial functional, concurrency, recovery and load evaluation; the eleven blockers in full. |
| [apple_silicon/ROADMAP.md](apple_silicon/ROADMAP.md) | What is done, what is next, and the definition of done. |
| [apple_silicon/COST_COMPARISON.md](apple_silicon/COST_COMPARISON.md) | What serving a RAG index costs on a Mac mini versus AWS versus Pinecone. |
| [apple_silicon/BASELINE.md](apple_silicon/BASELINE.md) | Lab notebook of the HNSW optimization campaign, including its corrections. |
| [apple_silicon/SEARCHLAYER_DECOMPOSITION.md](apple_silicon/SEARCHLAYER_DECOMPOSITION.md) | Profile of the index-build hot path and the hypotheses it ruled out. |
| [apple_silicon/PRIOR_EFFORT_AUDIT.md](apple_silicon/PRIOR_EFFORT_AUDIT.md) | Audit of the first porting attempt and what it actually proved. |
| [../scripts/bench/README.md](../scripts/bench/README.md) | The benchmark driver, the fairness contract, and how to A/B a change. |
| [../tools/apple_silicon/native_hnsw_smoke/README.md](../tools/apple_silicon/native_hnsw_smoke/README.md) | The standalone harness that compiles the production HNSW code. |
| [../test/eval_macos/README.md](../test/eval_macos/README.md) | The end-to-end evaluation suites and how to run them. |
| [../demo/README.md](../demo/README.md) | The hybrid-search demo and why its results come out the way they do. |
