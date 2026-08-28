#!/usr/bin/env python3
"""Extract ingestion-only evidence from a macOS `sample` call tree."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CALL_GRAPH_START = "Call graph:"
CALL_GRAPH_END = "Total number in stack"
LINE_RE = re.compile(r"^[ +:|!]*?(\d+) (.+)$")


@dataclass
class Node:
    count: int
    raw_symbol: str
    symbol: str
    depth: int
    line_number: int
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None

    @property
    def exclusive(self) -> int:
        return self.count - sum(child.count for child in self.children)


def canonical_symbol(raw_symbol: str) -> str:
    return raw_symbol.split("  (in ", 1)[0]


def parse_call_graph(path: Path) -> list[Node]:
    roots: list[Node] = []
    stack: list[Node] = []
    in_call_graph = False

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.rstrip("\n")
            if line == CALL_GRAPH_START:
                in_call_graph = True
                continue
            if in_call_graph and line.startswith(CALL_GRAPH_END):
                break
            if not in_call_graph:
                continue

            match = LINE_RE.match(line)
            if match is None:
                continue
            count_column = match.start(1)
            if count_column < 4 or (count_column - 4) % 2:
                raise ValueError(
                    f"unexpected count column {count_column + 1} on line {line_number}"
                )
            depth = (count_column - 4) // 2
            raw_symbol = match.group(2)
            node = Node(
                count=int(match.group(1)),
                raw_symbol=raw_symbol,
                symbol=canonical_symbol(raw_symbol),
                depth=depth,
                line_number=line_number,
            )

            if depth == 0:
                roots.append(node)
                stack = [node]
                continue
            if not stack or depth > len(stack):
                raise ValueError(f"invalid depth transition on line {line_number}")
            stack = stack[:depth]
            parent = stack[-1]
            node.parent = parent
            parent.children.append(node)
            stack.append(node)

    if not roots:
        raise ValueError("no call graph roots found")
    return roots


def iter_nodes(nodes: Iterable[Node]) -> Iterable[Node]:
    for node in nodes:
        yield node
        yield from iter_nodes(node.children)


def terminal_paths(
    node: Node, ancestors: tuple[Node, ...] = ()
) -> Iterable[tuple[int, tuple[Node, ...]]]:
    path = ancestors + (node,)
    if node.exclusive:
        yield node.exclusive, path
    for child in node.children:
        yield from terminal_paths(child, path)


def find_thread(node: Node) -> Node:
    while node.parent is not None:
        node = node.parent
    return node


def is_task_root(node: Node) -> bool:
    return (
        "std::__packaged_task_func" in node.symbol
        and "HnswRunBuildTasks" in node.symbol
    )


def classify_phase(symbols: set[str]) -> str:
    if any("ConnectNeighbors(" in symbol for symbol in symbols):
        return "reciprocal_connect_and_prune"
    if any("SelectNeighborsHeuristic<false>" in symbol for symbol in symbols):
        return "new_node_neighbor_selection"
    if any("SearchLayerNearest<" in symbol for symbol in symbols):
        return "upper_layer_entry_search"
    if any("SearchLayer<" in symbol for symbol in symbols):
        return "construction_graph_traversal"
    if any("BuildLocked(" in symbol for symbol in symbols):
        return "build_locked_other"
    if any("BuildWithOperationLockHeld(" in symbol for symbol in symbols):
        return "build_wrapper_other"
    return "task_wrapper_other"


def path_has(symbols: set[str], needles: tuple[str, ...]) -> bool:
    return any(needle in symbol for symbol in symbols for needle in needles)


def percentage(count: int, denominator: int) -> float:
    return 100.0 * count / denominator if denominator else math.nan


def ranked(counter: Counter[str], denominator: int, limit: int = 30) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "samples": count,
            "percent_of_task_active_samples": percentage(count, denominator),
        }
        for symbol, count in counter.most_common(limit)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    roots = parse_call_graph(args.sample_report)
    all_nodes = list(iter_nodes(roots))
    invalid_exclusive = [
        {
            "line": node.line_number,
            "symbol": node.symbol,
            "count": node.count,
            "child_sum": sum(child.count for child in node.children),
        }
        for node in all_nodes
        if node.exclusive < 0
    ]
    if invalid_exclusive:
        raise ValueError(f"negative exclusive counts: {invalid_exclusive[:3]}")

    task_roots = [node for node in all_nodes if is_task_root(node)]
    if len(task_roots) != 12:
        raise ValueError(f"expected 12 task roots, found {len(task_roots)}")

    task_roots.sort(key=lambda node: find_thread(node).line_number)
    task_counts = [node.count for node in task_roots]
    denominator = sum(task_counts)
    maximum_task_count = max(task_counts)

    exclusive_symbols: Counter[str] = Counter()
    inclusive_symbols: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    phase_distance_kernels: dict[str, Counter[str]] = defaultdict(Counter)
    stack_categories: Counter[str] = Counter()
    exclusive_source_lines: dict[str, set[int]] = defaultdict(set)
    terminal_total = 0

    for task_root in task_roots:
        root_terminal_total = 0
        for samples, path in terminal_paths(task_root):
            root_terminal_total += samples
            terminal_total += samples
            symbols = {node.symbol for node in path}
            endpoint = path[-1]
            phase = classify_phase(symbols)

            exclusive_symbols[endpoint.symbol] += samples
            exclusive_source_lines[endpoint.symbol].add(endpoint.line_number)
            phases[phase] += samples
            for symbol in sorted(symbols):
                inclusive_symbols[symbol] += samples

            if path_has(symbols, ("F32L2SSEBatch4@",)):
                phase_distance_kernels[phase]["batch4"] += samples
                stack_categories["batch4_distance"] += samples
            elif path_has(symbols, ("F32L2SSE@",)):
                phase_distance_kernels[phase]["scalar"] += samples
                stack_categories["scalar_distance"] += samples

            if path_has(symbols, ("__psynch_cvwait", "__psynch_mutexwait", "__ulock_wait")):
                stack_categories["blocked_lock_wait"] += samples
            if path_has(
                symbols,
                (
                    "std::__shared_mutex_base::",
                    "std::mutex::",
                    "pthread_mutex_",
                    "_pthread_mutex_",
                ),
            ):
                stack_categories["lock_path"] += samples
            if path_has(
                symbols,
                (
                    "operator new(",
                    "operator delete(",
                    "malloc",
                    "_xzm_",
                    "_free",
                    "free(",
                ),
            ):
                stack_categories["allocator_path"] += samples
            if path_has(symbols, ("SIMDPrefetch(",)):
                stack_categories["prefetch"] += samples
            if path_has(symbols, ("HeapResultHandler@",)):
                stack_categories["heap_result_handler"] += samples
            if path_has(symbols, ("std::vector<bool>::__construct_at_end",)):
                stack_categories["visited_bitmap_initialization"] += samples

        if root_terminal_total != task_root.count:
            raise ValueError(
                f"terminal accounting mismatch at line {task_root.line_number}: "
                f"{root_terminal_total} != {task_root.count}"
            )

    worker_rows = []
    for ordinal, task_root in enumerate(task_roots):
        thread = find_thread(task_root)
        worker_rows.append(
            {
                "worker_ordinal_by_sample_thread_order": ordinal,
                "thread": thread.symbol,
                "thread_line": thread.line_number,
                "task_root_line": task_root.line_number,
                "active_samples": task_root.count,
                "percent_of_longest_worker": percentage(
                    task_root.count, maximum_task_count
                ),
                "tail_inactive_samples_vs_longest": maximum_task_count
                - task_root.count,
            }
        )

    source_line_rows = [
        {
            "symbol": row["symbol"],
            "sample_call_tree_lines": sorted(exclusive_source_lines[row["symbol"]]),
        }
        for row in ranked(exclusive_symbols, denominator)
    ]

    result = {
        "schema": 1,
        "input": str(args.sample_report),
        "methodology": {
            "filter": (
                "Only subtrees rooted at the 12 std::__packaged_task_func frames "
                "whose symbols contain HnswRunBuildTasks."
            ),
            "exclusive": (
                "For each node, inclusive count minus direct-child counts; "
                "terminal counts sum exactly to each task-root count."
            ),
            "inclusive": (
                "For each terminal stack, samples are added once per distinct "
                "canonical symbol on that path, preventing recursive-frame double counting."
            ),
            "occupancy": (
                "sum(worker task-root samples) / (12 * max worker task-root samples); "
                "this is sampled active-stack occupancy, not hardware CPU utilization."
            ),
        },
        "validation": {
            "call_graph_roots": len(roots),
            "call_graph_nodes": len(all_nodes),
            "task_roots": len(task_roots),
            "task_threads": len({find_thread(node).symbol for node in task_roots}),
            "negative_exclusive_nodes": len(invalid_exclusive),
            "terminal_samples": terminal_total,
            "terminal_matches_task_root_sum": terminal_total == denominator,
        },
        "worker_occupancy": {
            "task_active_sample_observations": denominator,
            "worker_count": len(task_roots),
            "minimum_active_samples": min(task_counts),
            "maximum_active_samples": maximum_task_count,
            "mean_active_samples": denominator / len(task_roots),
            "minimum_to_maximum_ratio": min(task_counts) / maximum_task_count,
            "sample_window_active_stack_occupancy": denominator
            / (len(task_roots) * maximum_task_count),
            "tail_capacity_sample_observations": len(task_roots)
            * maximum_task_count
            - denominator,
            "workers": worker_rows,
        },
        "phase_exclusive_samples": [
            {
                "phase": phase,
                "samples": samples,
                "percent_of_task_active_samples": percentage(samples, denominator),
            }
            for phase, samples in phases.most_common()
        ],
        "distance_kernel_samples_by_phase": {
            phase: {
                kernel: {
                    "samples": samples,
                    "percent_of_task_active_samples": percentage(samples, denominator),
                }
                for kernel, samples in counts.items()
            }
            for phase, counts in sorted(phase_distance_kernels.items())
        },
        "stack_categories": [
            {
                "category": category,
                "samples": samples,
                "percent_of_task_active_samples": percentage(samples, denominator),
            }
            for category, samples in stack_categories.most_common()
        ],
        "top_exclusive_symbols": ranked(exclusive_symbols, denominator),
        "top_inclusive_symbols_deduplicated_per_stack": ranked(
            inclusive_symbols, denominator
        ),
        "top_exclusive_symbol_sample_lines": source_line_rows,
    }

    rendered = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
