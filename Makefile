# Infinity for Apple Silicon: the everyday commands.
#
# Every target is a thin wrapper over a script in scripts/apple_silicon/, so you can
# always run the script directly to see what a target does or to pass more options.
# `make` with no arguments prints this list.

SHELL := /bin/bash
.DEFAULT_GOAL := help

SCRIPTS  := scripts/apple_silicon
PRESET   ?= macos-arm64-release
INSTANCE ?= dev
UV       ?= uv
SERVER   := build/$(PRESET)/src/infinity
TEST_BIN := build/$(PRESET)/src/test_main

# The build needs the macOS SDK path; export it for every recipe so nobody has to.
export SDKROOT ?= $(shell xcrun --show-sdk-path 2>/dev/null)

# Threads for the benchmark. The published table used 10 (a 10-core M4); your own
# core count gives your machine's number rather than a comparison against the table.
PARTICIPANTS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || echo 8)

.PHONY: help doctor setup build build-tests start stop status restart logs demo sdk \
        test slt recovery eval package bench-build bench-faiss bench-datasets bench \
        lint clean-instances

help: ## Show this list
	@awk 'BEGIN {FS = ":.*## "; printf "\nUsage: make <target> [INSTANCE=name] [PRESET=macos-arm64-release]\n"} \
	      /^##@/ {printf "\n%s\n", substr($$0, 5)} \
	      /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\n'

##@ Getting started
doctor: ## Check prerequisites; changes nothing
	@$(SCRIPTS)/doctor.sh

setup: ## Install the toolchain, fetch dependencies and build the server (10 to 15 minutes on an M4)
	@$(SCRIPTS)/setup.sh

build: ## Rebuild the server after a code change
	@$(SCRIPTS)/build_server.sh $(PRESET) infinity

build-tests: ## Build the server and the unit-test binary
	@$(SCRIPTS)/build_server.sh $(PRESET) infinity test_main

##@ Running
start: ## Start a server on 127.0.0.1 (ports 23817 SDK, 23820 HTTP, 5432 Postgres)
	@$(SCRIPTS)/run_server.sh start --instance $(INSTANCE)

stop: ## Stop the server
	@$(SCRIPTS)/run_server.sh stop --instance $(INSTANCE)

status: ## Is the server running?
	@$(SCRIPTS)/run_server.sh status --instance $(INSTANCE)

restart: ## Restart the server
	@$(SCRIPTS)/run_server.sh restart --instance $(INSTANCE)

logs: ## Follow the server log
	@tail -n 50 -f build/instances/$(INSTANCE)/log/infinity.log

demo: ## Run the hybrid-search demo against the running server
	@$(UV) run --with fastembed demo/hybrid_search_demo.py

sdk: ## Install the Python SDK and test dependencies into .venv
	@$(UV) sync --python 3.11 --all-extras

##@ Testing
test: ## Run the C++ unit tests (after make build-tests)
	@test -x $(TEST_BIN) || { echo "no $(TEST_BIN); run: make build-tests"; exit 1; }
	@$(TEST_BIN)

slt: ## Run the SQL logic tests on a fresh, separate instance
	@$(SCRIPTS)/install_test_tools.sh
	@$(SCRIPTS)/run_server.sh start --instance ci --fresh
	@$(UV) run python $(SCRIPTS)/run_slt.py; status=$$?; $(SCRIPTS)/run_server.sh stop --instance ci; exit $$status

recovery: ## Kill a server mid-write and check every acknowledged commit comes back
	@$(SCRIPTS)/verify_crash_recovery.sh --rows 2000

eval: ## Run the end-to-end evaluation suites in test/eval_macos, one server each
	@$(UV) run python test/eval_macos/run_parallel_suite.py

package: ## Build a relocatable tarball and start it from a scratch directory to prove it works
	@$(SCRIPTS)/make_package.sh

##@ Benchmarks (independent of the server build; no vcpkg needed)
bench-build: ## Build the HNSW benchmark harness (a few minutes)
	@$(SCRIPTS)/bootstrap_ctpl.sh
	@cmake --preset bench -S tools/apple_silicon/native_hnsw_smoke
	@cmake --build build/bench --target infinity_hnsw_d0 faiss_hnsw_d0

bench-faiss: ## Build the Accelerate-linked FAISS reference (the fair comparison)
	@$(SCRIPTS)/build_faiss_accelerate.sh

bench-datasets: ## Download SIFT1M into ./datasets (168 MB)
	@python3 scripts/bench/fetch_datasets.py sift1m

bench: ## Paired Infinity vs FAISS run on SIFT1M (about 15 minutes)
	@test -x build/bench/infinity_hnsw_d0 || { echo "harness not built; run: make bench-build"; exit 1; }
	@test -x build/bench-faiss-src/faiss_hnsw_d0 || { echo "the Accelerate-linked FAISS reference is not built; run: make bench-faiss"; echo "(the Homebrew FAISS in build/bench is 1.5x slower and would flatter Infinity, so it is never used here)"; exit 1; }
	@test -f datasets/sift1m/base.f32 || { echo "dataset missing; run: make bench-datasets"; exit 1; }
	@HNSW_D0_EXTERNAL_QUERIES=$$PWD/datasets/sift1m/query.f32 \
	 HNSW_D0_EXTERNAL_GROUNDTRUTH=$$PWD/datasets/sift1m/groundtruth.i32 \
	 python3 scripts/bench/run_baseline.py \
	   --dataset datasets/sift1m/base.f32 --n 1000000 --d 128 --m 32 --efc 200 \
	   --ef 32,64,128,256 --participants $(PARTICIPANTS) --pairs 2 \
	   --infinity-bin build/bench/infinity_hnsw_d0 --faiss-bin build/bench-faiss-src/faiss_hnsw_d0

##@ Housekeeping
lint: ## Shellcheck the scripts and check every relative link in the Markdown
	@if command -v shellcheck >/dev/null; then shellcheck -S warning $(SCRIPTS)/*.sh install.sh; else echo "shellcheck not installed (brew install shellcheck); skipping"; fi
	@python3 scripts/check_md_links.py

clean-instances: ## Stop every local server instance and delete its data (build/instances)
	@for d in build/instances/*/; do [ -d "$$d" ] || continue; $(SCRIPTS)/run_server.sh stop --instance "$$(basename "$$d")" || true; done
	@rm -rf build/instances
