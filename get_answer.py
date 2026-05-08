"""
Run all tasks in a prompts.json category over images; save JSON results.

Usage:
    export OPENAI_API_KEY="your_key"
    python get_answer.py --category "Periapical Radiographs" --from-category-dirs \\
        --output results/pa_run.json
    python get_answer.py --category "Panoramic Radiographs" --image-dir dataset/PANO2 \\
        --output results/pano.json

Optional:
    export OPENAI_BASE_URL="https://api.openai.com/v1"
    export OPENAI_BASE_URL="https://api.minimax.io/v1"
    # Note: https://api.minimax.com often does not resolve (DNS). Use api.minimax.io per MiniMax docs.
    python get_answer.py ... --task "Caries Detection"

Local vLLM (LLaVA etc.): use ./run_model.sh or set OPENAI_BASE_URL and run with --vllm
(see run_model.sh). Example: OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://127.0.0.1:8000/v1
python get_answer.py --vllm --model llava-1.5-7b --category ... -o out.json
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None  # type: ignore

T = TypeVar("T")

# Three imaging branches (prompts.json top-level keys). Paths are relative to repo root.
DATASET_DIRS_BY_CATEGORY = {
    "Periapical Radiographs": ["dataset/PA4", "dataset/PA5"],
    "Panoramic Radiographs": ["dataset/PANO2"],
    "Lateral Cephalometric Radiographs": ["dataset/CEPH2"],
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

MINIMAX_THINKING_BLOCK = re.compile(
    r"<redacted_(reasoning|thinking)>[\s\S]*?</redacted_\1>\s*",
    re.IGNORECASE,
)


def strip_minimax_visible_answer(text: str) -> str:
    """Remove MiniMax redacted-reasoning wrapper from assistant text when present."""
    if not text:
        return text
    return MINIMAX_THINKING_BLOCK.sub("", text).strip()


def normalize_openai_base_url(url: str | None) -> str | None:
    """MiniMax docs use api.minimax.io; api.minimax.com often fails DNS or is outdated."""
    if not url or not url.strip():
        return url
    original = url.strip().rstrip("/")
    rewritten = re.sub(
        r"api\.minimax\.com",
        "api.minimax.io",
        original,
        flags=re.IGNORECASE,
    )
    if rewritten != original:
        print(
            f"Normalized OPENAI_BASE_URL for MiniMax: {original} -> {rewritten}",
            flush=True,
        )
    return rewritten


def format_error_short(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    c = exc.__cause__
    depth = 0
    while c is not None and depth < 3:
        parts.append(f"{type(c).__name__}: {c}")
        c = getattr(c, "__cause__", None)
        depth += 1
    return " || ".join(parts)


def to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_output_text(resp) -> str:
    # OpenAI Responses API
    output_text = getattr(resp, "output_text", "")
    if output_text:
        return output_text.strip()

    # Chat Completions (OpenAI-style / Zhipu zai-sdk)
    choices = getattr(resp, "choices", None) or []
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            # vLLM / some OpenAI-compatible servers use a list of content parts
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") in {"text", "output_text"}:
                            t = block.get("text", "")
                            if isinstance(t, str) and t:
                                parts.append(t)
                joined = "\n".join(parts).strip()
                if joined:
                    return joined

    # Responses API nested fallback
    texts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) in {"output_text", "text"}:
                t = getattr(c, "text", "")
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()


def collect_images_from_directory(directory: Path, *, recursive: bool) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    found: list[Path] = []
    for p in iterator:
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            found.append(p)
    return sorted(found)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _is_dns_or_host_not_found(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "nodename nor servname" in msg
        or "name or service not known" in msg
        or "failed to resolve" in msg
        or "name resolution" in msg
    )


def _is_retryable_api_error(exc: Exception) -> bool:
    if _is_dns_or_host_not_found(exc):
        return False
    if isinstance(
        exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ):
        return True
    code = getattr(exc, "status_code", None)
    resp = getattr(exc, "response", None)
    if code is None and resp is not None:
        code = getattr(resp, "status_code", None)
    if code in {429, 500, 502, 503, 529}:
        return True
    msg = str(exc).lower()
    if "overloaded" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    return False


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay_s: float,
) -> T:
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if _is_dns_or_host_not_found(e):
                print(
                    "Hint: hostname did not resolve. If you use MiniMax, set\n"
                    '  OPENAI_BASE_URL="https://api.minimax.io/v1"\n'
                    "(not api.minimax.com). Retrying will not fix DNS.",
                    flush=True,
                )
                raise
            if attempt >= max_retries or not _is_retryable_api_error(e):
                raise
            delay = min(base_delay_s * (2**attempt), 120.0)
            ra = _retry_after_seconds(e)
            if ra is not None:
                delay = max(delay, ra)
            print(
                f"... transient API error, sleep {delay:.1f}s then retry "
                f"(try {attempt + 2}/{max_retries + 1}): {format_error_short(e)}",
                flush=True,
            )
            time.sleep(delay)
    assert last is not None
    raise last


def main() -> None:
    parser = argparse.ArgumentParser(description="Test image+text API time response.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="Image file path (repeat for multiple).",
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory whose image files are included (repeat for multiple).",
    )
    parser.add_argument(
        "--recursive-image-dir",
        action="store_true",
        help="With --image-dir, scan subdirectories too (rglob).",
    )
    parser.add_argument(
        "--from-category-dirs",
        action="store_true",
        help="Add images from DATASET_DIRS_BY_CATEGORY paths for --category (relative to repo root).",
    )
    parser.add_argument(
        "--prompts-json",
        type=Path,
        default=Path(__file__).resolve().parent / "prompts.json",
        help="Path to prompts.json (default: next to this script).",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="prompts.json top-level key (e.g. Periapical Radiographs). All tasks in it are run unless --task.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="If set, run only this task within --category (must match prompts.json key).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Write results JSON to this path (parent dirs are created if needed).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="Model name (default from OPENAI_MODEL or gpt-4.1-mini).",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help=(
            "Target a local vLLM OpenAI-compatible server: if OPENAI_BASE_URL is unset, "
            "use http://127.0.0.1:8000/v1; if OPENAI_API_KEY is unset, use EMPTY."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Optional max_tokens for chat.completions (recommended for some local VLMs).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retries for transient errors (429, 5xx, connection resets). DNS errors are not retried.",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="Initial backoff delay; doubles each retry (capped). Default 5.",
    )
    args = parser.parse_args()

    prompts_path = args.prompts_json.expanduser().resolve()
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    with prompts_path.open(encoding="utf-8") as f:
        prompts_data: dict[str, dict[str, str]] = json.load(f)
    
    dependency_path = args.dependency_json.expanduser().resolve()
    if not dependency_path.is_file():
        raise FileNotFoundError(f"Dependency file not found: {dependency_path}")
    with dependency_path.open(encoding="utf-8") as f:
        dependency_data: dict[str, dict[str, list[str]]] = json.load(f)

    category = args.category
    if category not in prompts_data:
        parser.error(f"Unknown category in prompts.json: {category!r}")

    task_names = list(prompts_data[category].keys())
    if args.task is not None:
        if args.task not in prompts_data[category]:
            parser.error(
                f"Unknown task {args.task!r} for category {category!r}. "
                f"Valid: {task_names}"
            )
        tasks_to_run = [args.task]
    else:
        tasks_to_run = task_names

    repo_root = Path(__file__).resolve().parent
    image_paths: list[Path] = []
    for raw in args.image:
        image_paths.append(Path(raw).expanduser().resolve())
    for raw in args.image_dir:
        image_paths.extend(
            collect_images_from_directory(
                Path(raw).expanduser().resolve(),
                recursive=args.recursive_image_dir,
            )
        )
    if args.from_category_dirs:
        if category not in DATASET_DIRS_BY_CATEGORY:
            parser.error(
                f"--from-category-dirs: category {category!r} has no paths in DATASET_DIRS_BY_CATEGORY."
            )
        for rel in DATASET_DIRS_BY_CATEGORY[category]:
            image_paths.extend(
                collect_images_from_directory(
                    (repo_root / rel).resolve(),
                    recursive=args.recursive_image_dir,
                )
            )

    image_paths = dedupe_paths(image_paths)
    if not image_paths:
        parser.error(
            "No images: pass --image (repeatable), --image-dir (repeatable), "
            "and/or --from-category-dirs."
        )

    if args.model == "glm-4.6v":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set OPENAI_API_KEY first.")
        client = ZhipuAiClient(api_key=api_key)
        openai_base: str | None = None
    else:
        if args.vllm:
            api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
            openai_base = normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))
            if not openai_base:
                openai_base = "http://127.0.0.1:8000/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Please set OPENAI_API_KEY first.")
            openai_base = normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))
        client = OpenAI(
            api_key=api_key,
            base_url=openai_base,
        )

    results_rows: list[dict] = []
    latencies_s: list[float] = []
    latencies_by_task: dict[str, list[float]] = {t: [] for t in tasks_to_run}

    total_pairs = len(image_paths) * len(tasks_to_run)
    pair_idx = 0
    for image_path in image_paths:
        if not image_path.is_file():
            print(f"\nSKIP (not a file): {image_path}")
            for task_name in tasks_to_run:
                pair_idx += 1
                results_rows.append(
                    {
                        "image": str(image_path.resolve()),
                        "category": category,
                        "task": task_name,
                        "skipped": True,
                        "error": "not a file",
                        "response": "",
                        "latency_ms": None,
                    }
                )
            continue

        image_data_url = to_data_url(image_path)
        
        saved_results = dict()

        for task_name in tasks_to_run:
            pair_idx += 1
            prompt_text = prompts_data[category][task_name]
            dependency_tasks = dependency_data[category][task_name]
            if dependency_tasks is not None:
                prompt_text += "Additional information:\n"
                for dependency_task in dependency_tasks:
                    if dependency_task not in saved_results:
                        print(f"Dependency task {dependency_task} not found in saved results")
                        continue
                    prompt_text += f"{dependency_task}: {saved_results[dependency_task]}\n"
                prompt_text += "\n"
            
            print(
                f"\n{'=' * 60}\n[{pair_idx}/{total_pairs}] {image_path}  |  task: {task_name}\n{'=' * 60}"
            )

            user_prefix = "You are a dental radiologist. Please answer the question briefly and directly.\n\n"
            api_elapsed_s = 0.0

            def do_api_call():
                nonlocal api_elapsed_s
                t_req = time.perf_counter()
                if args.model == "glm-4.6v":
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prefix + prompt_text},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_data_url},
                                    },
                                ],
                            }
                        ],
                        temperature=0.0,
                    )
                elif args.model == "gpt-5.2":
                    r = client.responses.create(
                        model=args.model,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": user_prefix + prompt_text,
                                    },
                                    {
                                        "type": "input_image",
                                        "image_url": image_data_url,
                                    },
                                ],
                            }
                        ],
                    )
                else:
                    # OpenAI-compatible chat (MiniMax, vLLM, OpenAI vision, etc.)
                    cc_kwargs: dict = {
                        "model": args.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    # {"type": "text", "text": user_prefix + prompt_text},
                                    {"type": "text", "text": "What is in the image?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_data_url},
                                    },
                                ],
                            }
                        ],
                        "temperature": 0.0,
                    }
                    if args.max_tokens is not None:
                        cc_kwargs["max_tokens"] = args.max_tokens
                    # MiniMax: optional `reasoning_split` only moves thinking to `reasoning_details`;
                    # it does not disable generation. Strip <think> in post-processing.
                    if openai_base and "minimax.io" in openai_base.lower():
                        cc_kwargs["extra_body"] = {"reasoning_split": True}
                    r = client.chat.completions.create(**cc_kwargs)
                api_elapsed_s = time.perf_counter() - t_req
                return r

            response = call_with_retries(
                do_api_call,
                max_retries=args.max_retries,
                base_delay_s=args.retry_base_seconds,
            )
            elapsed_s = api_elapsed_s
            latencies_s.append(elapsed_s)
            latencies_by_task[task_name].append(elapsed_s)

            text = extract_output_text(response)
            text = strip_minimax_visible_answer(text)

            print("=== Latency (request → full response) ===")
            print(f"{elapsed_s * 1000:.2f} ms ({elapsed_s:.3f} s)")
            print("\n=== API Response ===")
            print(text or "<empty>")

            saved_results[task_name] = text

            results_rows.append(
                {
                    "image": str(image_path.resolve()),
                    "category": category,
                    "task": task_name,
                    "skipped": False,
                    "error": None,
                    "response": text,
                    "latency_ms": round(elapsed_s * 1000, 3),
                }
            )

    if latencies_s:
        n = len(latencies_s)
        mean_s = sum(latencies_s) / n
        min_s = min(latencies_s)
        max_s = max(latencies_s)
        print(f"\n{'=' * 60}\n=== Latency summary ({n} API call(s)) ===")
        print(
            f"mean: {mean_s * 1000:.2f} ms ({mean_s:.3f} s)  "
            f"min: {min_s * 1000:.2f} ms  max: {max_s * 1000:.2f} ms"
        )

    latency_by_task_summary: dict[str, dict[str, float]] = {}
    for t_name, samples in latencies_by_task.items():
        if not samples:
            continue
        latency_by_task_summary[t_name] = {
            "n": len(samples),
            "mean_ms": round(sum(samples) / len(samples) * 1000, 3),
            "min_ms": round(min(samples) * 1000, 3),
            "max_ms": round(max(samples) * 1000, 3),
        }

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "category": category,
            "tasks_run": tasks_to_run,
            "prompts_json": str(prompts_path),
            "image_count": len(image_paths),
            "api_calls": len(latencies_s),
        },
        "latency_summary_ms": (
            {
                "n": len(latencies_s),
                "mean": round(mean_s * 1000, 3),
                "min": round(min_s * 1000, 3),
                "max": round(max_s * 1000, 3),
            }
            if latencies_s
            else None
        ),
        "latency_by_task_ms": latency_by_task_summary,
        "results": results_rows,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n=== Saved results ===\n{out_path}")


if __name__ == "__main__":
    # GPT-5.2, GLM-4.6, Kimi-K2 (kimi-k2.5), Qwen-Plus, ABAB-6.5, LLaVA-7B
    main()
