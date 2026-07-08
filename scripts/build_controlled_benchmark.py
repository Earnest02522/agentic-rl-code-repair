#!/usr/bin/env python3
"""Build controlled mini-repo code repair tasks runnable without Docker."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TemplateTask:
    task_id: str
    difficulty: str
    bug_family: str
    instruction: str
    source_path: str
    buggy_source: str
    fixed_source: str
    test_source: str


def clean(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def run(cmd: list[str], cwd: Path, timeout: int = 30) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
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
            "@@ -1,{old_len} +1,{new_len} @@".format(old_len=len(old_lines), new_len=len(new_lines)),
            *["-" + line for line in old_lines],
            *["+" + line for line in new_lines],
            "",
        ]
    )


def templates() -> list[TemplateTask]:
    tasks: list[TemplateTask] = []

    def add(
        task_id: str,
        difficulty: str,
        bug_family: str,
        instruction: str,
        source_path: str,
        buggy: str,
        fixed: str,
        test: str,
    ) -> None:
        tasks.append(
            TemplateTask(
                task_id=task_id,
                difficulty=difficulty,
                bug_family=bug_family,
                instruction=instruction,
                source_path=source_path,
                buggy_source=clean(buggy),
                fixed_source=clean(fixed),
                test_source=clean(test),
            )
        )

    add(
        "py_parse_bool_001",
        "easy",
        "parsing_normalization",
        "Fix boolean parsing so common false strings are not treated as true.",
        "app/parsing.py",
        """
        def parse_bool(value):
            if isinstance(value, bool):
                return value
            return bool(str(value).strip())
        """,
        """
        def parse_bool(value):
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off", ""}:
                return False
            raise ValueError(f"invalid boolean value: {value!r}")
        """,
        """
        import pytest
        from app.parsing import parse_bool

        def test_false_strings():
            assert parse_bool("false") is False
            assert parse_bool("0") is False
            assert parse_bool("off") is False

        def test_true_strings():
            assert parse_bool("yes") is True
            assert parse_bool(True) is True

        def test_invalid_string():
            with pytest.raises(ValueError):
                parse_bool("maybe")
        """,
    )

    add(
        "py_config_merge_002",
        "medium",
        "config_merge",
        "Fix nested configuration merge so nested defaults are preserved.",
        "app/config.py",
        """
        def merge_config(defaults, overrides):
            merged = dict(defaults)
            merged.update(overrides)
            return merged
        """,
        """
        def merge_config(defaults, overrides):
            merged = dict(defaults)
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = merge_config(merged[key], value)
                else:
                    merged[key] = value
            return merged
        """,
        """
        from app.config import merge_config

        def test_nested_defaults_are_preserved():
            defaults = {"db": {"host": "localhost", "port": 5432}, "debug": False}
            overrides = {"db": {"port": 15432}}
            assert merge_config(defaults, overrides) == {
                "db": {"host": "localhost", "port": 15432},
                "debug": False,
            }
        """,
    )

    add(
        "py_window_average_003",
        "easy",
        "boundary_condition",
        "Fix moving average so the first windows use available history only.",
        "app/metrics.py",
        """
        def moving_average(values, window):
            result = []
            for index in range(len(values)):
                start = index - window
                chunk = values[start:index]
                result.append(sum(chunk) / len(chunk))
            return result
        """,
        """
        def moving_average(values, window):
            if window <= 0:
                raise ValueError("window must be positive")
            result = []
            for index in range(len(values)):
                start = max(0, index - window + 1)
                chunk = values[start:index + 1]
                result.append(sum(chunk) / len(chunk))
            return result
        """,
        """
        import pytest
        from app.metrics import moving_average

        def test_uses_available_history():
            assert moving_average([2, 4, 6, 8], 3) == [2, 3, 4, 6]

        def test_rejects_bad_window():
            with pytest.raises(ValueError):
                moving_average([1], 0)
        """,
    )

    add(
        "py_path_join_004",
        "easy",
        "path_handling",
        "Fix path normalization so absolute child paths cannot escape the root.",
        "app/paths.py",
        """
        from pathlib import Path

        def safe_join(root, child):
            return str(Path(root) / child)
        """,
        """
        from pathlib import Path

        def safe_join(root, child):
            root_path = Path(root).resolve()
            child_path = (root_path / child).resolve()
            if child_path != root_path and root_path not in child_path.parents:
                raise ValueError("child path escapes root")
            return str(child_path)
        """,
        """
        import pytest
        from app.paths import safe_join

        def test_relative_child_is_allowed(tmp_path):
            assert safe_join(tmp_path, "a/b.txt").endswith("a/b.txt")

        def test_parent_escape_is_rejected(tmp_path):
            with pytest.raises(ValueError):
                safe_join(tmp_path, "../secret.txt")
        """,
    )

    add(
        "py_cache_key_005",
        "medium",
        "cache_invalidation",
        "Fix the cache key so keyword arguments affect cached results.",
        "app/cache.py",
        """
        def memoize(fn):
            cache = {}
            def wrapper(*args, **kwargs):
                if args not in cache:
                    cache[args] = fn(*args, **kwargs)
                return cache[args]
            return wrapper
        """,
        """
        def memoize(fn):
            cache = {}
            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                if key not in cache:
                    cache[key] = fn(*args, **kwargs)
                return cache[key]
            return wrapper
        """,
        """
        from app.cache import memoize

        def test_kwargs_are_part_of_cache_key():
            calls = []

            @memoize
            def scale(value, factor=1):
                calls.append((value, factor))
                return value * factor

            assert scale(3, factor=2) == 6
            assert scale(3, factor=4) == 12
            assert scale(3, factor=2) == 6
            assert calls == [(3, 2), (3, 4)]
        """,
    )

    add(
        "py_dedupe_tags_006",
        "easy",
        "normalization_deduplication",
        "Fix tag normalization so tags are trimmed, lowercased, and deduplicated in first-seen order.",
        "app/tags.py",
        """
        def normalize_tags(tags):
            return [tag.strip() for tag in tags if tag.strip()]
        """,
        """
        def normalize_tags(tags):
            normalized = []
            seen = set()
            for tag in tags:
                value = tag.strip().lower()
                if value and value not in seen:
                    normalized.append(value)
                    seen.add(value)
            return normalized
        """,
        """
        from app.tags import normalize_tags

        def test_tags_are_normalized_and_deduped():
            assert normalize_tags([" API ", "api", "", "Bug", " bug "]) == ["api", "bug"]
        """,
    )

    add(
        "py_group_counts_007",
        "easy",
        "aggregation",
        "Fix event counting so repeated event types are accumulated instead of overwritten.",
        "app/events.py",
        """
        def count_events(events):
            counts = {}
            for event in events:
                counts[event["type"]] = 1
            return counts
        """,
        """
        def count_events(events):
            counts = {}
            for event in events:
                event_type = event["type"]
                counts[event_type] = counts.get(event_type, 0) + 1
            return counts
        """,
        """
        from app.events import count_events

        def test_counts_repeated_event_types():
            events = [{"type": "click"}, {"type": "view"}, {"type": "click"}]
            assert count_events(events) == {"click": 2, "view": 1}
        """,
    )

    add(
        "py_top_k_008",
        "medium",
        "sorting_ranking",
        "Fix top-k ranking so larger scores come first and ties are sorted by item id.",
        "app/ranking.py",
        """
        def top_k(items, k):
            return sorted(items, key=lambda item: item["score"])[:k]
        """,
        """
        def top_k(items, k):
            if k <= 0:
                return []
            return sorted(items, key=lambda item: (-item["score"], item["id"]))[:k]
        """,
        """
        from app.ranking import top_k

        def test_top_k_descending_with_stable_tie_break():
            items = [
                {"id": "b", "score": 9},
                {"id": "a", "score": 9},
                {"id": "c", "score": 4},
            ]
            assert top_k(items, 2) == [{"id": "a", "score": 9}, {"id": "b", "score": 9}]

        def test_non_positive_k_returns_empty():
            assert top_k([{"id": "x", "score": 1}], 0) == []
        """,
    )

    add(
        "py_email_validate_009",
        "easy",
        "validation",
        "Fix email validation so malformed addresses with missing local or domain parts are rejected.",
        "app/email_utils.py",
        """
        def is_valid_email(email):
            return "@" in email
        """,
        """
        def is_valid_email(email):
            if email.count("@") != 1:
                return False
            local, domain = email.split("@")
            return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")
        """,
        """
        from app.email_utils import is_valid_email

        def test_valid_email():
            assert is_valid_email("dev@example.com") is True

        def test_rejects_malformed_addresses():
            assert is_valid_email("@example.com") is False
            assert is_valid_email("dev@example") is False
            assert is_valid_email("a@b@c.com") is False
        """,
    )

    add(
        "py_slugify_010",
        "medium",
        "string_transformation",
        "Fix slug generation so punctuation and repeated separators are normalized.",
        "app/slug.py",
        """
        def slugify(text):
            return text.lower().replace(" ", "-")
        """,
        """
        import re

        def slugify(text):
            slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            return re.sub(r"-+", "-", slug)
        """,
        """
        from app.slug import slugify

        def test_slugify_normalizes_punctuation():
            assert slugify(" Hello, Agentic RL!! ") == "hello-agentic-rl"
            assert slugify("A---B") == "a-b"
        """,
    )

    add(
        "py_interval_overlap_011",
        "easy",
        "boundary_condition",
        "Fix interval overlap so touching half-open intervals are not counted as overlapping.",
        "app/intervals.py",
        """
        def overlaps(left, right):
            return left[0] <= right[1] and right[0] <= left[1]
        """,
        """
        def overlaps(left, right):
            return left[0] < right[1] and right[0] < left[1]
        """,
        """
        from app.intervals import overlaps

        def test_half_open_intervals():
            assert overlaps((1, 3), (3, 5)) is False
            assert overlaps((1, 4), (3, 5)) is True
        """,
    )

    add(
        "py_rate_limit_012",
        "medium",
        "state_machine",
        "Fix the rate limiter so old hits outside the rolling window are discarded.",
        "app/rate_limit.py",
        """
        class RateLimiter:
            def __init__(self, limit, window):
                self.limit = limit
                self.window = window
                self.hits = []

            def allow(self, timestamp):
                self.hits.append(timestamp)
                return len(self.hits) <= self.limit
        """,
        """
        class RateLimiter:
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
        """
        from app.rate_limit import RateLimiter

        def test_rolling_window_expires_old_hits():
            limiter = RateLimiter(limit=2, window=10)
            assert limiter.allow(0) is True
            assert limiter.allow(1) is True
            assert limiter.allow(2) is False
            assert limiter.allow(11) is True
        """,
    )

    add(
        "py_cart_total_013",
        "easy",
        "numeric_aggregation",
        "Fix cart total calculation so quantity is included for each line item.",
        "app/cart.py",
        """
        def cart_total(items):
            return sum(item["price"] for item in items)
        """,
        """
        def cart_total(items):
            return sum(item["price"] * item.get("quantity", 1) for item in items)
        """,
        """
        from app.cart import cart_total

        def test_cart_total_uses_quantity():
            items = [{"price": 3, "quantity": 2}, {"price": 5}]
            assert cart_total(items) == 11
        """,
    )

    add(
        "py_query_parse_014",
        "medium",
        "parsing",
        "Fix query parsing so URL-encoded values and repeated keys are handled correctly.",
        "app/query.py",
        """
        def parse_query(query):
            result = {}
            for part in query.split("&"):
                key, value = part.split("=")
                result[key] = value
            return result
        """,
        """
        from urllib.parse import parse_qs

        def parse_query(query):
            parsed = parse_qs(query, keep_blank_values=True)
            return {key: values[-1] for key, values in parsed.items()}
        """,
        """
        from app.query import parse_query

        def test_parse_query_decodes_and_uses_last_value():
            assert parse_query("q=agentic+rl&page=1&page=2") == {"q": "agentic rl", "page": "2"}

        def test_blank_value_is_preserved():
            assert parse_query("debug=") == {"debug": ""}
        """,
    )

    add(
        "py_inventory_015",
        "medium",
        "state_update",
        "Fix inventory reservation so failed reservations do not mutate stock.",
        "app/inventory.py",
        """
        def reserve(stock, sku, quantity):
            stock[sku] = stock.get(sku, 0) - quantity
            return stock[sku] >= 0
        """,
        """
        def reserve(stock, sku, quantity):
            available = stock.get(sku, 0)
            if quantity <= 0:
                raise ValueError("quantity must be positive")
            if available < quantity:
                return False
            stock[sku] = available - quantity
            return True
        """,
        """
        import pytest
        from app.inventory import reserve

        def test_failed_reservation_does_not_mutate_stock():
            stock = {"book": 2}
            assert reserve(stock, "book", 3) is False
            assert stock == {"book": 2}

        def test_rejects_non_positive_quantity():
            with pytest.raises(ValueError):
                reserve({"book": 2}, "book", 0)
        """,
    )

    add(
        "py_token_mask_016",
        "easy",
        "string_masking",
        "Fix token masking so short tokens are fully hidden and long tokens keep only their suffix.",
        "app/secrets.py",
        """
        def mask_token(token):
            return token[:4] + "..."
        """,
        """
        def mask_token(token):
            if len(token) <= 4:
                return "*" * len(token)
            return "*" * (len(token) - 4) + token[-4:]
        """,
        """
        from app.secrets import mask_token

        def test_mask_token_keeps_only_suffix():
            assert mask_token("abcdef1234") == "******1234"
            assert mask_token("abc") == "***"
        """,
    )

    add(
        "py_csv_rows_017",
        "medium",
        "csv_parsing",
        "Fix CSV row loading so quoted commas are parsed correctly.",
        "app/csv_rows.py",
        """
        def load_rows(text):
            lines = text.strip().splitlines()
            headers = lines[0].split(",")
            return [dict(zip(headers, line.split(","))) for line in lines[1:]]
        """,
        """
        import csv
        from io import StringIO

        def load_rows(text):
            reader = csv.DictReader(StringIO(text))
            return list(reader)
        """,
        """
        from app.csv_rows import load_rows

        def test_quoted_commas_are_preserved():
            text = 'name,note\\nAda,"hello, world"\\n'
            assert load_rows(text) == [{"name": "Ada", "note": "hello, world"}]
        """,
    )

    add(
        "py_date_range_018",
        "easy",
        "date_logic",
        "Fix date range generation so the end date is included.",
        "app/dates.py",
        """
        from datetime import timedelta

        def date_range(start, end):
            days = []
            current = start
            while current < end:
                days.append(current)
                current += timedelta(days=1)
            return days
        """,
        """
        from datetime import timedelta

        def date_range(start, end):
            days = []
            current = start
            while current <= end:
                days.append(current)
                current += timedelta(days=1)
            return days
        """,
        """
        from datetime import date
        from app.dates import date_range

        def test_date_range_includes_end_date():
            assert date_range(date(2026, 1, 1), date(2026, 1, 3)) == [
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 3),
            ]
        """,
    )

    add(
        "py_retry_policy_019",
        "medium",
        "retry_logic",
        "Fix retry execution so the initial attempt plus configured retries are all attempted.",
        "app/retry.py",
        """
        def run_with_retries(fn, retries):
            last_error = None
            for _ in range(retries):
                try:
                    return fn()
                except Exception as exc:
                    last_error = exc
            raise last_error
        """,
        """
        def run_with_retries(fn, retries):
            last_error = None
            for _ in range(retries + 1):
                try:
                    return fn()
                except Exception as exc:
                    last_error = exc
            raise last_error
        """,
        """
        from app.retry import run_with_retries

        def test_initial_attempt_plus_retries():
            attempts = []

            def flaky():
                attempts.append(1)
                if len(attempts) < 3:
                    raise RuntimeError("not yet")
                return "ok"

            assert run_with_retries(flaky, retries=2) == "ok"
            assert len(attempts) == 3
        """,
    )

    add(
        "py_pagination_020",
        "easy",
        "pagination",
        "Fix pagination so page numbers are one-based.",
        "app/pagination.py",
        """
        def paginate(items, page, per_page):
            start = page * per_page
            return items[start:start + per_page]
        """,
        """
        def paginate(items, page, per_page):
            if page < 1:
                raise ValueError("page must be one-based")
            start = (page - 1) * per_page
            return items[start:start + per_page]
        """,
        """
        import pytest
        from app.pagination import paginate

        def test_page_numbers_are_one_based():
            assert paginate([1, 2, 3, 4], page=1, per_page=2) == [1, 2]
            assert paginate([1, 2, 3, 4], page=2, per_page=2) == [3, 4]

        def test_rejects_page_zero():
            with pytest.raises(ValueError):
                paginate([1], page=0, per_page=10)
        """,
    )

    add(
        "py_user_filter_021",
        "easy",
        "filtering",
        "Fix active user filtering so disabled users are excluded.",
        "app/users.py",
        """
        def active_users(users):
            return [user for user in users if user.get("last_login")]
        """,
        """
        def active_users(users):
            return [
                user
                for user in users
                if user.get("enabled") is True and user.get("last_login") is not None
            ]
        """,
        """
        from app.users import active_users

        def test_active_users_require_enabled_and_login():
            users = [
                {"id": 1, "enabled": True, "last_login": "today"},
                {"id": 2, "enabled": False, "last_login": "today"},
                {"id": 3, "enabled": True, "last_login": None},
            ]
            assert active_users(users) == [{"id": 1, "enabled": True, "last_login": "today"}]
        """,
    )

    add(
        "py_json_patch_022",
        "medium",
        "nested_update",
        "Fix nested dictionary updates so dotted keys create nested dictionaries.",
        "app/patching.py",
        """
        def set_value(data, key, value):
            data[key] = value
            return data
        """,
        """
        def set_value(data, key, value):
            parts = key.split(".")
            current = data
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
            return data
        """,
        """
        from app.patching import set_value

        def test_dotted_key_updates_nested_dict():
            data = {}
            assert set_value(data, "service.timeout", 30) == {"service": {"timeout": 30}}
        """,
    )

    add(
        "py_priority_queue_023",
        "medium",
        "heap_ordering",
        "Fix priority queue ordering so lower priority numbers are returned first.",
        "app/queueing.py",
        """
        import heapq

        class PriorityQueue:
            def __init__(self):
                self.items = []

            def push(self, priority, item):
                heapq.heappush(self.items, (-priority, item))

            def pop(self):
                return heapq.heappop(self.items)[1]
        """,
        """
        import heapq

        class PriorityQueue:
            def __init__(self):
                self.items = []

            def push(self, priority, item):
                heapq.heappush(self.items, (priority, item))

            def pop(self):
                return heapq.heappop(self.items)[1]
        """,
        """
        from app.queueing import PriorityQueue

        def test_lower_priority_number_pops_first():
            queue = PriorityQueue()
            queue.push(10, "later")
            queue.push(1, "now")
            assert queue.pop() == "now"
            assert queue.pop() == "later"
        """,
    )

    add(
        "py_discount_024",
        "easy",
        "business_rule",
        "Fix discount application so percent discounts are applied as percentages.",
        "app/discounts.py",
        """
        def apply_discount(price, percent):
            return price - percent
        """,
        """
        def apply_discount(price, percent):
            if not 0 <= percent <= 100:
                raise ValueError("percent must be between 0 and 100")
            return price * (1 - percent / 100)
        """,
        """
        import pytest
        from app.discounts import apply_discount

        def test_discount_is_percentage():
            assert apply_discount(200, 15) == 170

        def test_invalid_percent_is_rejected():
            with pytest.raises(ValueError):
                apply_discount(200, 150)
        """,
    )

    add(
        "py_schema_required_025",
        "easy",
        "schema_validation",
        "Fix schema validation so missing required fields are reported.",
        "app/schema.py",
        """
        def validate_required(data, required):
            return []
        """,
        """
        def validate_required(data, required):
            return [field for field in required if field not in data or data[field] in (None, "")]
        """,
        """
        from app.schema import validate_required

        def test_missing_required_fields_are_reported():
            assert validate_required({"name": "Ada", "email": ""}, ["name", "email", "role"]) == ["email", "role"]
        """,
    )

    add(
        "py_file_ext_026",
        "easy",
        "path_handling",
        "Fix extension extraction so hidden files are not treated as having extensions.",
        "app/file_names.py",
        """
        def extension(path):
            return path.split(".")[-1]
        """,
        """
        from pathlib import Path

        def extension(path):
            suffix = Path(path).suffix
            return suffix[1:] if suffix else ""
        """,
        """
        from app.file_names import extension

        def test_extension_handles_hidden_and_normal_files():
            assert extension("archive.tar.gz") == "gz"
            assert extension(".env") == ""
            assert extension("README") == ""
        """,
    )

    add(
        "py_stats_median_027",
        "medium",
        "statistics",
        "Fix median calculation so even-length inputs average the two middle values.",
        "app/stats.py",
        """
        def median(values):
            ordered = sorted(values)
            return ordered[len(ordered) // 2]
        """,
        """
        def median(values):
            if not values:
                raise ValueError("median requires at least one value")
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[mid]
            return (ordered[mid - 1] + ordered[mid]) / 2
        """,
        """
        import pytest
        from app.stats import median

        def test_median_for_even_and_odd_lengths():
            assert median([5, 1, 3]) == 3
            assert median([10, 2, 4, 8]) == 6

        def test_empty_values_rejected():
            with pytest.raises(ValueError):
                median([])
        """,
    )

    add(
        "py_markdown_title_028",
        "easy",
        "text_extraction",
        "Fix markdown title extraction so leading blank lines are ignored.",
        "app/markdown.py",
        """
        def extract_title(text):
            first = text.splitlines()[0]
            return first.lstrip("# ")
        """,
        """
        def extract_title(text):
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()
            return ""
        """,
        """
        from app.markdown import extract_title

        def test_extract_title_skips_leading_blank_lines():
            assert extract_title("\\n\\n# Project Two\\nBody") == "Project Two"
            assert extract_title("No title") == ""
        """,
    )

    add(
        "py_feature_flags_029",
        "medium",
        "precedence",
        "Fix feature flag lookup so user overrides take precedence over defaults.",
        "app/flags.py",
        """
        def is_enabled(name, defaults, overrides):
            return defaults.get(name, False) or overrides.get(name, False)
        """,
        """
        def is_enabled(name, defaults, overrides):
            if name in overrides:
                return bool(overrides[name])
            return bool(defaults.get(name, False))
        """,
        """
        from app.flags import is_enabled

        def test_override_false_takes_precedence():
            assert is_enabled("new_ui", {"new_ui": True}, {"new_ui": False}) is False
            assert is_enabled("search", {"search": False}, {"search": True}) is True
        """,
    )

    add(
        "py_batching_030",
        "easy",
        "chunking",
        "Fix batching so the final partial batch is retained.",
        "app/batching.py",
        """
        def batches(items, size):
            result = []
            for index in range(0, len(items) - size, size):
                result.append(items[index:index + size])
            return result
        """,
        """
        def batches(items, size):
            if size <= 0:
                raise ValueError("size must be positive")
            return [items[index:index + size] for index in range(0, len(items), size)]
        """,
        """
        import pytest
        from app.batching import batches

        def test_final_partial_batch_is_retained():
            assert batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

        def test_rejects_bad_batch_size():
            with pytest.raises(ValueError):
                batches([1], 0)
        """,
    )

    if len(tasks) != 30:
        raise RuntimeError(f"expected 30 controlled tasks, got {len(tasks)}")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("controlled task ids must be unique")
    return tasks


def write_task(root: Path, task: TemplateTask) -> dict[str, Any]:
    task_dir = root / task.task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "app").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "app" / "__init__.py").write_text("", encoding="utf-8")
    (task_dir / task.source_path).parent.mkdir(parents=True, exist_ok=True)
    (task_dir / task.source_path).write_text(task.buggy_source, encoding="utf-8")
    (task_dir / "tests" / "test_task.py").write_text(task.test_source, encoding="utf-8")
    require_ok(run(["git", "init"], cwd=task_dir), f"git init for {task.task_id}")
    require_ok(run(["git", "config", "user.email", "agentic-rl@example.local"], cwd=task_dir), f"git config email for {task.task_id}")
    require_ok(run(["git", "config", "user.name", "Agentic RL Builder"], cwd=task_dir), f"git config name for {task.task_id}")
    require_ok(run(["git", "add", "."], cwd=task_dir), f"git add for {task.task_id}")
    require_ok(run(["git", "commit", "-m", "initial buggy task"], cwd=task_dir), f"git commit for {task.task_id}")

    patch = simple_patch(task.buggy_source, task.fixed_source, task.source_path)
    return {
        "task_id": task.task_id,
        "instance_id": task.task_id,
        "repo": "controlled/python-mini-repair",
        "language": "Python",
        "difficulty": task.difficulty,
        "split": "controlled_v0",
        "source_dataset": "controlled_py_v0",
        "base_commit": "local",
        "problem_statement": task.instruction,
        "allowed_files": [task.source_path],
        "patch": patch,
        "test_patch": None,
        "test_command": "PYTHONPATH=. python -m pytest -q",
        "changed_files": 1,
        "changed_lines": sum(1 for line in patch.splitlines() if line.startswith("+") or line.startswith("-")) - 2,
        "workspace": str(task_dir),
        "metadata": {"bug_family": task.bug_family},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default="agentic-rl/workspaces/controlled_py_v0")
    parser.add_argument("--output", default="agentic-rl/data/tasks/controlled_py_v0.jsonl")
    args = parser.parse_args()

    root = Path(args.workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [write_task(root, task) for task in templates()]
    write_jsonl(Path(args.output), rows)

    results = []
    for row in rows:
        workspace = Path(row["workspace"])
        before_runs = [run(["bash", "-lc", row["test_command"]], cwd=workspace) for _ in range(2)]
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=workspace,
            input=row["patch"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        after_runs = [run(["bash", "-lc", row["test_command"]], cwd=workspace) for _ in range(2)]
        require_ok(run(["git", "reset", "--hard", "HEAD"], cwd=workspace), f"git reset for {row['task_id']}")
        results.append({
            "task_id": row["task_id"],
            "bug_family": row["metadata"]["bug_family"],
            "difficulty": row["difficulty"],
            "buggy_fails": all(not item["ok"] for item in before_runs),
            "patch_applies": proc.returncode == 0,
            "gold_passes": all(item["ok"] for item in after_runs),
            "before_exit_codes": [item["exit_code"] for item in before_runs],
            "after_exit_codes": [item["exit_code"] for item in after_runs],
            "patch_stderr": proc.stderr[-1000:],
        })

    bug_families = sorted({row["metadata"]["bug_family"] for row in rows})
    difficulties = {
        difficulty: sum(1 for row in rows if row["difficulty"] == difficulty)
        for difficulty in sorted({row["difficulty"] for row in rows})
    }
    report = {
        "tasks": len(rows),
        "buggy_fail": sum(1 for r in results if r["buggy_fails"]),
        "patch_apply": sum(1 for r in results if r["patch_applies"]),
        "gold_pass": sum(1 for r in results if r["gold_passes"]),
        "all_quality_gates_pass": all(
            r["buggy_fails"] and r["patch_applies"] and r["gold_passes"] for r in results
        ),
        "bug_families": bug_families,
        "difficulties": difficulties,
        "results": results,
    }
    Path("agentic-rl/results/controlled_py_v0_build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    main()
