# FAQ

**What is this, in one sentence?**
A native macOS build of the Infinity hybrid-search database, with its vector index
tuned for Apple Silicon, so you can run vector plus full-text plus SQL search on a Mac
without Docker.

**Is it a Pinecone, Qdrant, Chroma or Weaviate alternative?**
For a single machine, yes: it stores embeddings, searches them, and also does BM25 and
structured filters in the same query. It is not a hosted service and has no
multi-tenancy, authentication or TLS; put it behind your own service if you expose it.

**Why macOS only? Can I run it on Linux?**
Upstream [Infinity](https://github.com/infiniflow/infinity) runs on Linux and is the
right choice there. This fork removed the Linux and x86-64 paths so that nothing here
has to stay portable, and the build refuses a non-Darwin host on purpose.

**Does it run on an M1? An Intel Mac?**
Any Apple Silicon Mac on macOS 14 or newer. Releases are generic arm64 binaries with
runtime-dispatched kernels, so an M1 runs the same package as an M4. Intel Macs are not
supported; Rosetta cannot run it either.

**Is it production ready?**
No. It builds, runs and passes its suites, and its concurrency held up under stress,
but an adversarial evaluation found eleven defects, four of which let a client crash
the server and several of which return wrong answers with a success code.
[Known issues](known-issues.md) has the list and workarounds.

**Does it create embeddings?**
No. You run an embedding model (fastembed, sentence-transformers, OpenAI, anything)
and store the vectors. The [demo](../demo/) and the [RAG tutorial](guides/rag.md) use
fastembed because it runs on the CPU without PyTorch.

**How much fits on a 16 GB Mac?**
About three million 768-dimension vectors or one and a half million at 1,536; the rule
of thumb is `N × (4 × dimension + 256)` bytes. [Tuning](guides/tuning.md) has the table.

**How fast is it?**
On an M4 Mac mini: a 1,000,000-vector HNSW index in 35 seconds, about 25,000 queries
per second in-process at 128 dimensions, and about 5,000 of each at 768 dimensions.
The [README table](../README.md#benchmarks) and
[BENCHMARKS.md](apple_silicon/BENCHMARKS.md) state every parameter behind those numbers.

**Why does the build take an hour?**
It compiles about 1,500 C++23 module translation units plus thirty vcpkg
dependencies, single-machine. A prebuilt package from the releases page avoids it
entirely; `install.sh` fetches and verifies one.

**Can I use the `infinity-sdk` from PyPI?**
Yes. The server speaks the same protocol as upstream 0.7.3, so the published SDK works
against it. The copy in this repository has one extra fix (the RAG tokenizer finds its
dictionary from a checkout), which only matters if you use that tokenizer client-side.

**Why is there no embedded (in-process) mode?**
Upstream's embedded module is a native Python extension that this port does not build.
`infinity.connect()` takes a network address only; the commented-out
`connect("/var/infinity")` lines in the upstream examples are dead here.

**What does "hybrid search" mean and why should I care?**
Keyword search and vector search fail in opposite ways: one misses documents that use
different words, the other blurs rare exact terms. Running both and fusing the ranks
gets both kinds of question right. [Concepts](concepts.md) explains it; the demo shows
it with numbers.

**Where does my data go?**
From a checkout, under `build/instances/<name>/`. From a package, under
`~/.local/share/infinity` unless you pass `--data-dir`. [Operations](guides/operations.md)
has the full map.

**Does it work with RAGFlow?**
RAGFlow uses Infinity as one of its document engines and lists Infinity on ARM64 as
unsupported. Wiring this native build into an ARM64 RAGFlow setup is on the
[roadmap](apple_silicon/ROADMAP.md) and not done yet.

**Will this be contributed upstream?**
The intent is to re-land the platform work behind portability gates as reviewable
pull requests against infiniflow/infinity. The fork dropped Linux to move faster, so
that is real work rather than a cherry-pick, and it is on the roadmap.

**What is the license?**
Apache-2.0, the same as upstream. Infinity is © InfiniFlow; the derivative-work
attribution is in [NOTICE](../NOTICE).

**Where do I ask questions?**
[GitHub Discussions](https://github.com/jatinsethi98/infinity-apple-silicon/discussions)
for this port. For Infinity itself, upstream's
[documentation](https://infiniflow.org/docs/dev/) and
[Discord](https://discord.gg/jEfRUwEYEV).
