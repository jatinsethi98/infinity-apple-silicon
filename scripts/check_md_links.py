#!/usr/bin/env python3
"""Check every relative link and anchor in the repository's Markdown files.

Offline on purpose: CI should not go red because some external site is slow, so
http(s) links are listed but never fetched. What is checked:

  * a relative link's target exists (file or directory), relative to the file
  * a `#fragment` on a relative link, or a bare `#fragment`, matches a heading
    in the target file, using GitHub's anchor rules (lowercase, spaces to
    hyphens, punctuation dropped, duplicates suffixed -1, -2, ...)

Usage:
    python3 scripts/check_md_links.py [PATH ...]

With no arguments it walks the repository, skipping build output, vendored code
and the upstream reference docs it does not own. Exit status is 1 if anything
is broken, so it can run in CI and under `make lint`.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git", ".claude", "build", "third_party", "vcpkg_installed", "resource",
    "node_modules", ".venv", "datasets", "cmake-build-debug", "cmake-build-release",
    "gui", "python/benchmark", "docs/references", "test/data",
}

LINK_RE = re.compile(r"(?<!\!)\[(?:[^\]]|\]\[)*?\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
HTML_ANCHOR_RE = re.compile(r"<a\s+(?:name|id)=\"([^\"]+)\"")


def github_anchor(text: str) -> str:
    # Strip inline code/links/emphasis markers, then apply GitHub's slug rules.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = unicodedata.normalize("NFKD", text).lower().strip()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors_of(path: Path) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in HTML_ANCHOR_RE.finditer(line):
            anchors.add(m.group(1))
        m = HEADING_RE.match(line)
        if not m:
            continue
        slug = github_anchor(m.group(2))
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    return anchors


def markdown_files(args: list[str]) -> list[Path]:
    if args:
        out: list[Path] = []
        for a in args:
            p = Path(a)
            if p.is_dir():
                out.extend(sorted(f for f in p.rglob("*") if f.suffix in (".md", ".mdx")))
            else:
                out.append(p)
        return out
    files: list[Path] = []
    for root, dirs, names in os.walk(REPO):
        rel = Path(root).relative_to(REPO)
        dirs[:] = sorted(
            d for d in dirs
            if d not in SKIP_DIRS and str(rel / d) not in SKIP_DIRS and not d.startswith(".")
        )
        files.extend(Path(root) / n for n in sorted(names) if n.endswith((".md", ".mdx")))
    return files


def check(path: Path, anchor_cache: dict[Path, set[str]]) -> list[str]:
    problems: list[str] = []
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK_RE.finditer(line):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            frag = None
            if "#" in target:
                target, frag = target.split("#", 1)
            if target == "":
                dest = path
            else:
                dest = (path.parent / target).resolve()
                if not dest.exists():
                    problems.append(f"{path.relative_to(REPO)}:{lineno}: missing target {target}")
                    continue
            if frag is not None and frag != "" and dest.is_file() and dest.suffix in (".md", ".mdx"):
                if dest not in anchor_cache:
                    anchor_cache[dest] = anchors_of(dest)
                if frag.lower() not in anchor_cache[dest]:
                    where = "" if dest == path else f" in {dest.relative_to(REPO)}"
                    problems.append(f"{path.relative_to(REPO)}:{lineno}: no heading for #{frag}{where}")
    return problems


def main(argv: list[str]) -> int:
    files = markdown_files(argv)
    cache: dict[Path, set[str]] = {}
    problems: list[str] = []
    for f in files:
        problems.extend(check(f, cache))
    for p in problems:
        print(p)
    print(f"{len(files)} markdown files checked, {len(problems)} broken link(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
