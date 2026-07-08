#!/usr/bin/env python3
"""Build controlled_py_v2_800 with public tests and external hidden tests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "workspaces" / "controlled_py_v2_800"
DEFAULT_HIDDEN_ROOT = PROJECT_ROOT / "data" / "hidden_tests" / "controlled_py_v2_800"
DEFAULT_TASKS = PROJECT_ROOT / "data" / "tasks" / "controlled_py_v2_800.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "controlled_py_v2_800_build_report.json"
DEFAULT_MD_REPORT = PROJECT_ROOT / "reports" / "22_controlled_py_v2_800_report.md"


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    split: str
    difficulty: str
    bug_family: str
    instruction: str
    source_path: str
    buggy_source: str
    fixed_source: str
    public_test_source: str
    hidden_test_source: str
    family_index: int
    variant_index: int


def clean(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def run(cmd: list[str], cwd: Path, timeout: int = 30, input_text: str | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def require_ok(result: dict[str, Any], action: str) -> None:
    if not result["ok"]:
        raise RuntimeError(f"{action} failed\nstdout:\n{result['stdout']}\nstderr:\n{result['stderr']}")


def simple_patch(old: str, new: str, path: str) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            f"--- a/{path}",
            f"+++ b/{path}",
            f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@",
            *["-" + line for line in old_lines],
            *["+" + line for line in new_lines],
            "",
        ]
    )


def split_for_variant(variant: int) -> str:
    if variant <= 29:
        return "train"
    if variant <= 34:
        return "val"
    return "heldout_test"


def module_path(family_index: int, stem: str, variant: int) -> tuple[str, str]:
    module = f"f{family_index:02d}_{stem}_{variant:03d}"
    return f"app/{module}.py", f"app.{module}"


def make_task(
    *,
    family_index: int,
    variant: int,
    stem: str,
    difficulty: str,
    bug_family: str,
    instruction: str,
    buggy: str,
    fixed: str,
    public: str,
    hidden: str,
) -> GeneratedTask:
    source_path, _ = module_path(family_index, stem, variant)
    return GeneratedTask(
        task_id=f"controlled_py_v2_f{family_index:02d}_{stem}_{variant:03d}",
        split=split_for_variant(variant),
        difficulty=difficulty,
        bug_family=bug_family,
        instruction=instruction,
        source_path=source_path,
        buggy_source=clean(buggy),
        fixed_source=clean(fixed),
        public_test_source=clean(public),
        hidden_test_source=clean(hidden),
        family_index=family_index,
        variant_index=variant,
    )


def gen_parse_bool(i: int) -> GeneratedTask:
    path, mod = module_path(1, "parse_bool", i)
    fn = f"parse_bool_{i:03d}"
    return make_task(
        family_index=1,
        variant=i,
        stem="parse_bool",
        difficulty="easy",
        bug_family="parsing_normalization",
        instruction="Fix boolean parsing so common false strings are not treated as true.",
        buggy=f"""
        def {fn}(value):
            if isinstance(value, bool):
                return value
            return bool(str(value).strip())
        """,
        fixed=f"""
        def {fn}(value):
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {{"1", "true", "yes", "y", "on"}}:
                return True
            if text in {{"0", "false", "no", "n", "off", ""}}:
                return False
            raise ValueError(f"invalid boolean value: {{value!r}}")
        """,
        public=f"""
        import pytest
        from {mod} import {fn}

        def test_public_false_values():
            assert {fn}("false") is False
            assert {fn}("0") is False

        def test_public_true_values():
            assert {fn}("yes") is True

        def test_public_invalid_value():
            with pytest.raises(ValueError):
                {fn}("maybe-{i}")
        """,
        hidden=f"""
        import pytest
        from {mod} import {fn}

        def test_hidden_false_values():
            assert {fn}(" off ") is False
            assert {fn}("") is False

        def test_hidden_invalid_value():
            with pytest.raises(ValueError):
                {fn}("enabled")
        """,
    )


def gen_merge_config(i: int) -> GeneratedTask:
    _, mod = module_path(2, "merge_config", i)
    fn = f"merge_config_{i:03d}"
    port = 15000 + i
    return make_task(
        family_index=2,
        variant=i,
        stem="merge_config",
        difficulty="medium",
        bug_family="config_merge",
        instruction="Fix nested configuration merge so nested defaults are preserved.",
        buggy=f"""
        def {fn}(defaults, overrides):
            merged = dict(defaults)
            merged.update(overrides)
            return merged
        """,
        fixed=f"""
        def {fn}(defaults, overrides):
            merged = dict(defaults)
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {fn}(merged[key], value)
                else:
                    merged[key] = value
            return merged
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_nested_defaults_are_preserved():
            defaults = {{"db": {{"host": "localhost", "port": 5432}}, "debug": False}}
            overrides = {{"db": {{"port": {port}}}}}
            assert {fn}(defaults, overrides) == {{
                "db": {{"host": "localhost", "port": {port}}},
                "debug": False,
            }}
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_deeper_nested_defaults_are_preserved():
            defaults = {{"service": {{"retry": {{"count": 3, "delay": 1}}, "name": "api"}}}}
            overrides = {{"service": {{"retry": {{"delay": {i + 2}}}}}}}
            assert {fn}(defaults, overrides) == {{
                "service": {{"retry": {{"count": 3, "delay": {i + 2}}}, "name": "api"}}
            }}
        """,
    )


def gen_moving_average(i: int) -> GeneratedTask:
    _, mod = module_path(3, "moving_average", i)
    fn = f"moving_average_{i:03d}"
    base = i + 2
    return make_task(
        family_index=3,
        variant=i,
        stem="moving_average",
        difficulty="easy",
        bug_family="boundary_condition",
        instruction="Fix moving average so each point includes the current value and available history only.",
        buggy=f"""
        def {fn}(values, window):
            result = []
            for index in range(len(values)):
                start = index - window
                chunk = values[start:index]
                result.append(sum(chunk) / len(chunk))
            return result
        """,
        fixed=f"""
        def {fn}(values, window):
            if window <= 0:
                raise ValueError("window must be positive")
            result = []
            for index in range(len(values)):
                start = max(0, index - window + 1)
                chunk = values[start:index + 1]
                result.append(sum(chunk) / len(chunk))
            return result
        """,
        public=f"""
        import pytest
        from {mod} import {fn}

        def test_public_available_history():
            assert {fn}([{base}, {base + 2}, {base + 4}, {base + 6}], 3) == [{base}, {base + 1}, {base + 2}, {base + 4}]

        def test_public_bad_window():
            with pytest.raises(ValueError):
                {fn}([1], 0)
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_window_one_returns_original_values():
            assert {fn}([3, 6, 9], 1) == [3, 6, 9]
        """,
    )


def gen_safe_join(i: int) -> GeneratedTask:
    _, mod = module_path(4, "safe_join", i)
    fn = f"safe_join_{i:03d}"
    return make_task(
        family_index=4,
        variant=i,
        stem="safe_join",
        difficulty="easy",
        bug_family="path_handling",
        instruction="Fix path normalization so child paths cannot escape the configured root.",
        buggy=f"""
        from pathlib import Path

        def {fn}(root, child):
            return str(Path(root) / child)
        """,
        fixed=f"""
        from pathlib import Path

        def {fn}(root, child):
            root_path = Path(root).resolve()
            child_path = (root_path / child).resolve()
            if child_path != root_path and root_path not in child_path.parents:
                raise ValueError("child path escapes root")
            return str(child_path)
        """,
        public=f"""
        import pytest
        from {mod} import {fn}

        def test_public_relative_child_allowed(tmp_path):
            assert {fn}(tmp_path, "a/file-{i}.txt").endswith("a/file-{i}.txt")

        def test_public_parent_escape_rejected(tmp_path):
            with pytest.raises(ValueError):
                {fn}(tmp_path, "../secret.txt")
        """,
        hidden=f"""
        import pytest
        from {mod} import {fn}

        def test_hidden_absolute_escape_rejected(tmp_path):
            with pytest.raises(ValueError):
                {fn}(tmp_path, "/etc/passwd")
        """,
    )


def gen_memoize(i: int) -> GeneratedTask:
    _, mod = module_path(5, "memoize", i)
    fn = f"memoize_{i:03d}"
    return make_task(
        family_index=5,
        variant=i,
        stem="memoize",
        difficulty="medium",
        bug_family="cache_invalidation",
        instruction="Fix the memoization cache key so keyword arguments affect cached results.",
        buggy=f"""
        def {fn}(func):
            cache = {{}}
            def wrapper(*args, **kwargs):
                if args not in cache:
                    cache[args] = func(*args, **kwargs)
                return cache[args]
            return wrapper
        """,
        fixed=f"""
        def {fn}(func):
            cache = {{}}
            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                if key not in cache:
                    cache[key] = func(*args, **kwargs)
                return cache[key]
            return wrapper
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_kwargs_part_of_cache_key():
            calls = []

            @{fn}
            def scale(value, factor=1):
                calls.append((value, factor))
                return value * factor

            assert scale({i + 2}, factor=2) == {(i + 2) * 2}
            assert scale({i + 2}, factor=4) == {(i + 2) * 4}
            assert calls == [({i + 2}, 2), ({i + 2}, 4)]
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_different_kwargs_do_not_share_cache_entry():
            calls = []

            @{fn}
            def shift(value, *, delta=0):
                calls.append(delta)
                return value + delta

            assert shift(10, delta=1) == 11
            assert shift(10, delta=3) == 13
            assert calls == [1, 3]
        """,
    )


def gen_tags(i: int) -> GeneratedTask:
    _, mod = module_path(6, "tags", i)
    fn = f"normalize_tags_{i:03d}"
    return make_task(
        family_index=6,
        variant=i,
        stem="tags",
        difficulty="easy",
        bug_family="normalization_deduplication",
        instruction="Fix tag normalization so tags are trimmed, lowercased, and deduplicated in first-seen order.",
        buggy=f"""
        def {fn}(tags):
            return [tag.strip() for tag in tags if tag.strip()]
        """,
        fixed=f"""
        def {fn}(tags):
            normalized = []
            seen = set()
            for tag in tags:
                value = tag.strip().lower()
                if value and value not in seen:
                    normalized.append(value)
                    seen.add(value)
            return normalized
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_tags_normalized_and_deduped():
            assert {fn}([" API ", "api", "", "Bug", " bug "]) == ["api", "bug"]
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_first_seen_order_is_preserved():
            assert {fn}(["Z{i}", " z{i} ", "Alpha"]) == ["z{i}", "alpha"]
        """,
    )


def gen_event_counts(i: int) -> GeneratedTask:
    _, mod = module_path(7, "event_counts", i)
    fn = f"count_events_{i:03d}"
    return make_task(
        family_index=7,
        variant=i,
        stem="event_counts",
        difficulty="easy",
        bug_family="aggregation",
        instruction="Fix event counting so repeated event types are accumulated instead of overwritten.",
        buggy=f"""
        def {fn}(events):
            counts = {{}}
            for event in events:
                counts[event["type"]] = 1
            return counts
        """,
        fixed=f"""
        def {fn}(events):
            counts = {{}}
            for event in events:
                event_type = event["type"]
                counts[event_type] = counts.get(event_type, 0) + 1
            return counts
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_counts_repeated_types():
            events = [{{"type": "click"}}, {{"type": "view"}}, {{"type": "click"}}]
            assert {fn}(events) == {{"click": 2, "view": 1}}
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_empty_and_three_repeats():
            assert {fn}([]) == {{}}
            assert {fn}([{{"type": "x"}}, {{"type": "x"}}, {{"type": "x"}}]) == {{"x": 3}}
        """,
    )


def gen_top_k(i: int) -> GeneratedTask:
    _, mod = module_path(8, "top_k", i)
    fn = f"top_k_{i:03d}"
    return make_task(
        family_index=8,
        variant=i,
        stem="top_k",
        difficulty="medium",
        bug_family="sorting_ranking",
        instruction="Fix top-k ranking so larger scores come first and ties are sorted by item id.",
        buggy=f"""
        def {fn}(items, k):
            return sorted(items, key=lambda item: item["score"])[:k]
        """,
        fixed=f"""
        def {fn}(items, k):
            if k <= 0:
                return []
            return sorted(items, key=lambda item: (-item["score"], item["id"]))[:k]
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_descending_with_tie_break():
            items = [{{"id": "b", "score": 9}}, {{"id": "a", "score": 9}}, {{"id": "c", "score": 4}}]
            assert {fn}(items, 2) == [{{"id": "a", "score": 9}}, {{"id": "b", "score": 9}}]
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_descending_without_ties():
            items = [{{"id": "low", "score": 1}}, {{"id": "high", "score": 10}}, {{"id": "mid", "score": 5}}]
            assert {fn}(items, 2) == [{{"id": "high", "score": 10}}, {{"id": "mid", "score": 5}}]
        """,
    )


def gen_slugify(i: int) -> GeneratedTask:
    _, mod = module_path(9, "slugify", i)
    fn = f"slugify_{i:03d}"
    return make_task(
        family_index=9,
        variant=i,
        stem="slugify",
        difficulty="medium",
        bug_family="string_transformation",
        instruction="Fix slug generation so punctuation and repeated separators are normalized.",
        buggy=f"""
        def {fn}(text):
            return text.lower().replace(" ", "-")
        """,
        fixed=f"""
        import re

        def {fn}(text):
            slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            return re.sub(r"-+", "-", slug)
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_punctuation_normalized():
            assert {fn}(" Hello, Agentic RL {i}!! ") == "hello-agentic-rl-{i}"
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_repeated_separators_collapsed():
            assert {fn}("A---B___C") == "a-b-c"
        """,
    )


def gen_rate_limiter(i: int) -> GeneratedTask:
    _, mod = module_path(10, "rate_limiter", i)
    cls = f"RateLimiterV{i:03d}"
    return make_task(
        family_index=10,
        variant=i,
        stem="rate_limiter",
        difficulty="medium",
        bug_family="state_machine",
        instruction="Fix the rate limiter so old hits outside the rolling window are discarded.",
        buggy=f"""
        class {cls}:
            def __init__(self, limit, window):
                self.limit = limit
                self.window = window
                self.hits = []

            def allow(self, timestamp):
                self.hits.append(timestamp)
                return len(self.hits) <= self.limit
        """,
        fixed=f"""
        class {cls}:
            def __init__(self, limit, window):
                self.limit = limit
                self.window = window
                self.hits = []

            def allow(self, timestamp):
                cutoff = timestamp - self.window
                self.hits = [hit for hit in self.hits if hit > cutoff]
                if len(self.hits) >= self.limit:
                    return False
                self.hits.append(timestamp)
                return True
        """,
        public=f"""
        from {mod} import {cls}

        def test_public_rolling_window_expires_old_hits():
            limiter = {cls}(limit=2, window=10)
            assert limiter.allow(0) is True
            assert limiter.allow(1) is True
            assert limiter.allow(2) is False
            assert limiter.allow(11) is True
        """,
        hidden=f"""
        from {mod} import {cls}

        def test_hidden_rejected_hit_is_not_recorded():
            limiter = {cls}(limit=1, window=5)
            assert limiter.allow(10) is True
            assert limiter.allow(11) is False
            assert limiter.allow(16) is True
        """,
    )


def gen_query_parse(i: int) -> GeneratedTask:
    _, mod = module_path(11, "query_parse", i)
    fn = f"parse_query_{i:03d}"
    return make_task(
        family_index=11,
        variant=i,
        stem="query_parse",
        difficulty="medium",
        bug_family="parsing",
        instruction="Fix query parsing so URL-encoded values and repeated keys are handled correctly.",
        buggy=f"""
        def {fn}(query):
            result = {{}}
            for part in query.split("&"):
                key, value = part.split("=")
                result[key] = value
            return result
        """,
        fixed=f"""
        from urllib.parse import parse_qs

        def {fn}(query):
            parsed = parse_qs(query, keep_blank_values=True)
            return {{key: values[-1] for key, values in parsed.items()}}
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_decodes_and_uses_last_value():
            assert {fn}("q=agentic+rl&page=1&page=2") == {{"q": "agentic rl", "page": "2"}}
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_blank_value_preserved():
            assert {fn}("debug=&name=Ada%20Lovelace") == {{"debug": "", "name": "Ada Lovelace"}}
        """,
    )


def gen_inventory(i: int) -> GeneratedTask:
    _, mod = module_path(12, "inventory", i)
    fn = f"reserve_{i:03d}"
    return make_task(
        family_index=12,
        variant=i,
        stem="inventory",
        difficulty="medium",
        bug_family="state_update",
        instruction="Fix inventory reservation so failed reservations do not mutate stock.",
        buggy=f"""
        def {fn}(stock, sku, quantity):
            stock[sku] = stock.get(sku, 0) - quantity
            return stock[sku] >= 0
        """,
        fixed=f"""
        def {fn}(stock, sku, quantity):
            available = stock.get(sku, 0)
            if quantity <= 0:
                raise ValueError("quantity must be positive")
            if available < quantity:
                return False
            stock[sku] = available - quantity
            return True
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_failed_reservation_does_not_mutate_stock():
            stock = {{"book": 2}}
            assert {fn}(stock, "book", 3) is False
            assert stock == {{"book": 2}}
        """,
        hidden=f"""
        import pytest
        from {mod} import {fn}

        def test_hidden_rejects_non_positive_quantity():
            with pytest.raises(ValueError):
                {fn}({{"book": 2}}, "book", 0)
        """,
    )


def gen_csv_rows(i: int) -> GeneratedTask:
    _, mod = module_path(13, "csv_rows", i)
    fn = f"load_rows_{i:03d}"
    return make_task(
        family_index=13,
        variant=i,
        stem="csv_rows",
        difficulty="medium",
        bug_family="csv_parsing",
        instruction="Fix CSV row loading so quoted commas are parsed correctly.",
        buggy=f"""
        def {fn}(text):
            lines = text.strip().splitlines()
            headers = lines[0].split(",")
            return [dict(zip(headers, line.split(","))) for line in lines[1:]]
        """,
        fixed=f"""
        import csv
        from io import StringIO

        def {fn}(text):
            reader = csv.DictReader(StringIO(text))
            return list(reader)
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_quoted_commas_are_preserved():
            text = 'name,note\\nAda,"hello, world {i}"\\n'
            assert {fn}(text) == [{{"name": "Ada", "note": "hello, world {i}"}}]
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_multiple_rows():
            text = 'name,note\\nAda,"a,b"\\nGrace,"c,d"\\n'
            assert {fn}(text)[1] == {{"name": "Grace", "note": "c,d"}}
        """,
    )


def gen_pagination(i: int) -> GeneratedTask:
    _, mod = module_path(14, "pagination", i)
    fn = f"paginate_{i:03d}"
    return make_task(
        family_index=14,
        variant=i,
        stem="pagination",
        difficulty="easy",
        bug_family="pagination",
        instruction="Fix pagination so page numbers are one-based.",
        buggy=f"""
        def {fn}(items, page, per_page):
            start = page * per_page
            return items[start:start + per_page]
        """,
        fixed=f"""
        def {fn}(items, page, per_page):
            if page < 1:
                raise ValueError("page must be one-based")
            start = (page - 1) * per_page
            return items[start:start + per_page]
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_page_numbers_are_one_based():
            assert {fn}([1, 2, 3, 4], page=1, per_page=2) == [1, 2]
            assert {fn}([1, 2, 3, 4], page=2, per_page=2) == [3, 4]
        """,
        hidden=f"""
        import pytest
        from {mod} import {fn}

        def test_hidden_rejects_page_zero():
            with pytest.raises(ValueError):
                {fn}([1], page=0, per_page=10)
        """,
    )


def gen_nested_set(i: int) -> GeneratedTask:
    _, mod = module_path(15, "nested_set", i)
    fn = f"set_value_{i:03d}"
    return make_task(
        family_index=15,
        variant=i,
        stem="nested_set",
        difficulty="medium",
        bug_family="nested_update",
        instruction="Fix nested dictionary updates so dotted keys create nested dictionaries.",
        buggy=f"""
        def {fn}(data, key, value):
            data[key] = value
            return data
        """,
        fixed=f"""
        def {fn}(data, key, value):
            parts = key.split(".")
            current = data
            for part in parts[:-1]:
                current = current.setdefault(part, {{}})
            current[parts[-1]] = value
            return data
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_dotted_key_updates_nested_dict():
            data = {{}}
            assert {fn}(data, "service.timeout", {30 + i}) == {{"service": {{"timeout": {30 + i}}}}}
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_deeper_key_updates_nested_dict():
            data = {{"service": {{"name": "api"}}}}
            assert {fn}(data, "service.retry.count", 3) == {{"service": {{"name": "api", "retry": {{"count": 3}}}}}}
        """,
    )


def gen_median(i: int) -> GeneratedTask:
    _, mod = module_path(16, "median", i)
    fn = f"median_{i:03d}"
    return make_task(
        family_index=16,
        variant=i,
        stem="median",
        difficulty="medium",
        bug_family="statistics",
        instruction="Fix median calculation so even-length inputs average the two middle values and empty input is rejected.",
        buggy=f"""
        def {fn}(values):
            ordered = sorted(values)
            return ordered[len(ordered) // 2]
        """,
        fixed=f"""
        def {fn}(values):
            if not values:
                raise ValueError("median requires at least one value")
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[mid]
            return (ordered[mid - 1] + ordered[mid]) / 2
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_even_and_odd_lengths():
            assert {fn}([5, 1, 3]) == 3
            assert {fn}([10, 2, 4, 8]) == 6
        """,
        hidden=f"""
        import pytest
        from {mod} import {fn}

        def test_hidden_empty_rejected():
            with pytest.raises(ValueError):
                {fn}([])
        """,
    )


def gen_batches(i: int) -> GeneratedTask:
    _, mod = module_path(17, "batches", i)
    fn = f"batches_{i:03d}"
    return make_task(
        family_index=17,
        variant=i,
        stem="batches",
        difficulty="easy",
        bug_family="chunking",
        instruction="Fix batching so the final partial batch is retained.",
        buggy=f"""
        def {fn}(items, size):
            result = []
            for index in range(0, len(items) - size, size):
                result.append(items[index:index + size])
            return result
        """,
        fixed=f"""
        def {fn}(items, size):
            if size <= 0:
                raise ValueError("size must be positive")
            return [items[index:index + size] for index in range(0, len(items), size)]
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_final_partial_batch_retained():
            assert {fn}([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_exact_size_batch_is_retained():
            assert {fn}(["a", "b"], 2) == [["a", "b"]]
        """,
    )


def gen_active_users(i: int) -> GeneratedTask:
    _, mod = module_path(18, "active_users", i)
    fn = f"active_users_{i:03d}"
    return make_task(
        family_index=18,
        variant=i,
        stem="active_users",
        difficulty="easy",
        bug_family="filtering",
        instruction="Fix active user filtering so disabled users and users without login are excluded.",
        buggy=f"""
        def {fn}(users):
            return [user for user in users if user.get("last_login")]
        """,
        fixed=f"""
        def {fn}(users):
            return [
                user
                for user in users
                if user.get("enabled") is True and user.get("last_login") is not None
            ]
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_requires_enabled_and_login():
            users = [
                {{"id": 1, "enabled": True, "last_login": "today"}},
                {{"id": 2, "enabled": False, "last_login": "today"}},
                {{"id": 3, "enabled": True, "last_login": None}},
            ]
            assert {fn}(users) == [{{"id": 1, "enabled": True, "last_login": "today"}}]
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_empty_login_string_is_still_login_value():
            users = [{{"id": {i}, "enabled": True, "last_login": ""}}]
            assert {fn}(users) == users
        """,
    )


def gen_extension(i: int) -> GeneratedTask:
    _, mod = module_path(19, "extension", i)
    fn = f"extension_{i:03d}"
    return make_task(
        family_index=19,
        variant=i,
        stem="extension",
        difficulty="easy",
        bug_family="path_handling",
        instruction="Fix extension extraction so hidden files and files without suffix return an empty extension.",
        buggy=f"""
        def {fn}(path):
            return path.split(".")[-1]
        """,
        fixed=f"""
        from pathlib import Path

        def {fn}(path):
            suffix = Path(path).suffix
            return suffix[1:] if suffix else ""
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_extension_handles_hidden_and_normal_files():
            assert {fn}("archive.tar.gz") == "gz"
            assert {fn}(".env") == ""
        """,
        hidden=f"""
        from {mod} import {fn}

        def test_hidden_no_extension_returns_empty():
            assert {fn}("README") == ""
        """,
    )


def gen_discount(i: int) -> GeneratedTask:
    _, mod = module_path(20, "discount", i)
    fn = f"apply_discount_{i:03d}"
    price = 200 + i
    return make_task(
        family_index=20,
        variant=i,
        stem="discount",
        difficulty="easy",
        bug_family="business_rule",
        instruction="Fix discount application so percent discounts are applied as percentages.",
        buggy=f"""
        def {fn}(price, percent):
            return price - percent
        """,
        fixed=f"""
        def {fn}(price, percent):
            if not 0 <= percent <= 100:
                raise ValueError("percent must be between 0 and 100")
            return price * (1 - percent / 100)
        """,
        public=f"""
        from {mod} import {fn}

        def test_public_discount_is_percentage():
            assert {fn}({price}, 10) == {price * 0.9}
        """,
        hidden=f"""
        import pytest
        from {mod} import {fn}

        def test_hidden_invalid_percent_rejected():
            with pytest.raises(ValueError):
                {fn}(100, 150)
        """,
    )


GENERATORS: list[Callable[[int], GeneratedTask]] = [
    gen_parse_bool,
    gen_merge_config,
    gen_moving_average,
    gen_safe_join,
    gen_memoize,
    gen_tags,
    gen_event_counts,
    gen_top_k,
    gen_slugify,
    gen_rate_limiter,
    gen_query_parse,
    gen_inventory,
    gen_csv_rows,
    gen_pagination,
    gen_nested_set,
    gen_median,
    gen_batches,
    gen_active_users,
    gen_extension,
    gen_discount,
]


def generate_tasks() -> list[GeneratedTask]:
    tasks: list[GeneratedTask] = []
    for generator in GENERATORS:
        for variant in range(40):
            tasks.append(generator(variant))
    if len(tasks) != 800:
        raise RuntimeError(f"expected 800 tasks, got {len(tasks)}")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("task ids must be unique")
    return tasks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_task(workspace_root: Path, hidden_root: Path, task: GeneratedTask) -> dict[str, Any]:
    task_dir = workspace_root / task.task_id
    hidden_dir = hidden_root / task.task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    if hidden_dir.exists():
        shutil.rmtree(hidden_dir)
    (task_dir / "app").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)
    hidden_dir.mkdir(parents=True)

    (task_dir / "app" / "__init__.py").write_text("", encoding="utf-8")
    (task_dir / task.source_path).parent.mkdir(parents=True, exist_ok=True)
    (task_dir / task.source_path).write_text(task.buggy_source, encoding="utf-8")
    (task_dir / "tests" / "test_public.py").write_text(task.public_test_source, encoding="utf-8")
    hidden_test_path = hidden_dir / "test_hidden.py"
    hidden_test_path.write_text(task.hidden_test_source, encoding="utf-8")

    require_ok(run(["git", "init"], cwd=task_dir), f"git init for {task.task_id}")
    require_ok(run(["git", "config", "user.email", "agentic-rl@example.local"], cwd=task_dir), f"git config email for {task.task_id}")
    require_ok(run(["git", "config", "user.name", "Agentic RL Builder"], cwd=task_dir), f"git config name for {task.task_id}")
    require_ok(run(["git", "add", "."], cwd=task_dir), f"git add for {task.task_id}")
    require_ok(run(["git", "commit", "-m", "initial buggy task"], cwd=task_dir), f"git commit for {task.task_id}")

    patch = simple_patch(task.buggy_source, task.fixed_source, task.source_path)
    public_command = "PYTHONPATH=. python -m pytest -q tests"
    hidden_command = f"PYTHONPATH=. python -m pytest -q {hidden_test_path}"
    verifier_command = f"PYTHONPATH=. python -m pytest -q tests {hidden_test_path}"
    return {
        "task_id": task.task_id,
        "instance_id": task.task_id,
        "repo": "controlled/python-mini-repair-v2",
        "language": "Python",
        "difficulty": task.difficulty,
        "split": task.split,
        "source_dataset": "controlled_py_v2_800",
        "base_commit": "local",
        "problem_statement": task.instruction,
        "allowed_files": [task.source_path],
        "patch": patch,
        "test_patch": None,
        "test_command": public_command,
        "changed_files": 1,
        "changed_lines": sum(1 for line in patch.splitlines() if line.startswith("+") or line.startswith("-")) - 2,
        "workspace": str(task_dir),
        "metadata": {
            "bug_family": task.bug_family,
            "family_index": task.family_index,
            "variant_index": task.variant_index,
            "public_test_command": public_command,
            "hidden_test_command": hidden_command,
            "verifier_command": verifier_command,
            "hidden_test_file": str(hidden_test_path),
            "hidden_tests_in_workspace": False,
            "split_policy": "per_family_30_train_5_val_5_heldout",
        },
    }


def should_repeat_stability(row: dict[str, Any], repeat_all: bool, stability_sample: int) -> bool:
    if repeat_all:
        return True
    if stability_sample <= 0:
        return False
    if stability_sample >= len(GENERATORS) * 2:
        return row["metadata"]["variant_index"] in {0, 14}
    return row["metadata"]["variant_index"] == 0


def validate_rows(rows: list[dict[str, Any]], *, repeat_all: bool = False, stability_sample: int = 40) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows, start=1):
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"[validate] {index}/{len(rows)} {row['task_id']}", flush=True)
        workspace = Path(row["workspace"])
        public_command = row["metadata"]["public_test_command"]
        hidden_command = row["metadata"]["hidden_test_command"]
        verifier_command = row["metadata"]["verifier_command"]
        repeats = 2 if should_repeat_stability(row, repeat_all, stability_sample) else 1

        before_public = [run(["bash", "-lc", public_command], cwd=workspace) for _ in range(repeats)]
        before_hidden = [run(["bash", "-lc", hidden_command], cwd=workspace) for _ in range(repeats)]
        before_final = run(["bash", "-lc", verifier_command], cwd=workspace)
        patch_result = run(["git", "apply", "--whitespace=nowarn", "-"], cwd=workspace, input_text=row["patch"])
        after_public = [run(["bash", "-lc", public_command], cwd=workspace) for _ in range(repeats)]
        after_hidden = [run(["bash", "-lc", hidden_command], cwd=workspace) for _ in range(repeats)]
        after_final = [run(["bash", "-lc", verifier_command], cwd=workspace) for _ in range(repeats)]
        require_ok(run(["git", "reset", "--hard", "HEAD"], cwd=workspace), f"git reset for {row['task_id']}")

        results.append(
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "bug_family": row["metadata"]["bug_family"],
                "difficulty": row["difficulty"],
                "buggy_public_fails": all(not item["ok"] for item in before_public),
                "buggy_hidden_fails": all(not item["ok"] for item in before_hidden),
                "buggy_final_fails": not before_final["ok"],
                "patch_applies": patch_result["ok"],
                "gold_public_passes": all(item["ok"] for item in after_public),
                "gold_hidden_passes": all(item["ok"] for item in after_hidden),
                "gold_final_passes": all(item["ok"] for item in after_final),
                "stability_repeated": repeats > 1,
                "repeat_count": repeats,
                "before_public_exit_codes": [item["exit_code"] for item in before_public],
                "before_hidden_exit_codes": [item["exit_code"] for item in before_hidden],
                "after_public_exit_codes": [item["exit_code"] for item in after_public],
                "after_hidden_exit_codes": [item["exit_code"] for item in after_hidden],
                "patch_stderr": patch_result["stderr"],
            }
        )
    return results


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    splits = sorted({row["split"] for row in rows})
    return {split: sum(1 for row in rows if row["split"] == split) for split in splits}


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Controlled Python Mini-Repo Benchmark v2 800

Date: 2026-07-06

## Result

`controlled_py_v2_800` expands the initial 30-task benchmark to 800 deterministic Python code-repair tasks with separated public and hidden tests.

| Check | Result |
| --- | ---: |
| Total tasks | {report["tasks"]} |
| Train split | {report["splits"].get("train", 0)} |
| Val split | {report["splits"].get("val", 0)} |
| Held-out test split | {report["splits"].get("heldout_test", 0)} |
| Buggy public tests fail | {report["buggy_public_fail"]} / {report["tasks"]} |
| Buggy hidden tests fail | {report["buggy_hidden_fail"]} / {report["tasks"]} |
| Gold patch applies | {report["patch_apply"]} / {report["tasks"]} |
| Gold public tests pass | {report["gold_public_pass"]} / {report["tasks"]} |
| Gold hidden tests pass | {report["gold_hidden_pass"]} / {report["tasks"]} |
| Stability repeated samples | {report["stability_repeated"]} / {report["tasks"]} |
| All quality gates pass | {report["all_quality_gates_pass"]} |

## Files

```text
agentic-rl/data/tasks/controlled_py_v2_800.jsonl
agentic-rl/data/tasks/controlled_py_v2_800_train.jsonl
agentic-rl/data/tasks/controlled_py_v2_800_val.jsonl
agentic-rl/data/tasks/controlled_py_v2_800_heldout_test.jsonl
agentic-rl/data/hidden_tests/controlled_py_v2_800/
agentic-rl/workspaces/controlled_py_v2_800/
agentic-rl/results/controlled_py_v2_800_build_report.json
```

## Design

The benchmark contains 20 generation families with 40 variants per family. They map to 19 bug-family labels because two generation families are path-handling variants. Each generation family contributes 30 train tasks, 5 validation tasks, and 5 held-out test tasks, so the held-out set has 100 tasks and every generation family appears in held-out evaluation.

Public tests live inside each task workspace under `tests/` and are the tests an agent may run during repair. Hidden tests live outside the workspace under `agentic-rl/data/hidden_tests/controlled_py_v2_800/`, so the four code-exec tools cannot inspect them by relative workspace paths. The final verifier runs both public and hidden tests.

## Quality Gates

Each task is accepted only if:

1. buggy code fails public tests
2. buggy code fails hidden tests
3. gold patch applies with `git apply`
4. patched code passes public tests
5. patched code passes hidden tests
6. patched code passes final public+hidden verifier
7. 40 stability samples, two from each generation family, repeat the public/hidden/final checks for stability

This is the dataset to use for base rollout, SFT data generation, GRPO/RLVR reward computation, and held-out evaluation.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--hidden-root", default=str(DEFAULT_HIDDEN_ROOT))
    parser.add_argument("--tasks-out", default=str(DEFAULT_TASKS))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--md-report-out", default=str(DEFAULT_MD_REPORT))
    parser.add_argument("--stability-sample", type=int, default=40)
    parser.add_argument("--repeat-all", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root)
    hidden_root = Path(args.hidden_root)
    tasks_out = Path(args.tasks_out)
    report_out = Path(args.report_out)
    md_report_out = Path(args.md_report_out)

    workspace_root.mkdir(parents=True, exist_ok=True)
    hidden_root.mkdir(parents=True, exist_ok=True)

    tasks = generate_tasks()
    rows = [write_task(workspace_root, hidden_root, task) for task in tasks]

    write_jsonl(tasks_out, rows)
    for split in ["train", "val", "heldout_test"]:
        split_path = tasks_out.with_name(f"{tasks_out.stem}_{split}.jsonl")
        write_jsonl(split_path, [row for row in rows if row["split"] == split])

    results = validate_rows(rows, repeat_all=args.repeat_all, stability_sample=args.stability_sample)
    report = {
        "tasks": len(rows),
        "splits": split_counts(rows),
        "buggy_public_fail": sum(1 for item in results if item["buggy_public_fails"]),
        "buggy_hidden_fail": sum(1 for item in results if item["buggy_hidden_fails"]),
        "buggy_final_fail": sum(1 for item in results if item["buggy_final_fails"]),
        "patch_apply": sum(1 for item in results if item["patch_applies"]),
        "gold_public_pass": sum(1 for item in results if item["gold_public_passes"]),
        "gold_hidden_pass": sum(1 for item in results if item["gold_hidden_passes"]),
        "gold_final_pass": sum(1 for item in results if item["gold_final_passes"]),
        "stability_repeated": sum(1 for item in results if item["stability_repeated"]),
        "all_quality_gates_pass": all(
            item["buggy_public_fails"]
            and item["buggy_hidden_fails"]
            and item["buggy_final_fails"]
            and item["patch_applies"]
            and item["gold_public_passes"]
            and item["gold_hidden_passes"]
            and item["gold_final_passes"]
            for item in results
        ),
        "bug_families": sorted({row["metadata"]["bug_family"] for row in rows}),
        "difficulties": {
            difficulty: sum(1 for row in rows if row["difficulty"] == difficulty)
            for difficulty in sorted({row["difficulty"] for row in rows})
        },
        "results": results,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_report(md_report_out, report)
    print(json.dumps({k: report[k] for k in report if k != "results"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
