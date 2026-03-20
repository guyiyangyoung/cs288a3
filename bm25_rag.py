#!/usr/bin/env python3
"""
BM25-based baseline RAG entrypoint.

Reads questions from a text file, retrieves relevant chunks from a prebuilt
chunk artifact, generates concise answers through llm.py, and writes one
prediction per line.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi
from query_expansion import expand_query_bm25

from rag_common import (
    ALLOWED_MODELS,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_RETRIES,
    DEFAULT_TIMEOUT_RETRY_BACKOFF_SEC,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_TOP_K,
    SCRIPT_DIR,
    answer_question,
    atomic_write,
    configure_logging,
    load_chunks,
    resolve_path,
    validate_runtime_env,
)

LOG = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"\b\w+\b")
BM25_CACHE_VERSION = 1
TOKENIZER_NAME = "regex_word_lower_v1"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict], bm25: BM25Okapi):
        self.chunks = chunks
        self.backend_name = "rank_bm25"
        self.bm25 = bm25

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        query_tokens = expand_query_bm25(question)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[dict] = []
        seen_passages: set[tuple[str, str]] = set()
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            signature = (
                chunk.get("url", ""),
                chunk.get("raw_text", ""),
            )
            if signature in seen_passages:
                continue
            seen_passages.add(signature)
            results.append(chunk)
        return results


def get_bm25_cache_paths(chunks_path: Path, cache_path: Path | None = None) -> tuple[Path, Path]:
    if cache_path is None:
        cache_path = chunks_path.with_suffix(".bm25.pkl")
    metadata_path = cache_path.with_suffix(".meta.json")
    return cache_path, metadata_path


def build_bm25_cache_metadata(chunks_path: Path) -> dict[str, object]:
    stat = chunks_path.stat()
    return {
        "cache_version": BM25_CACHE_VERSION,
        "tokenizer": TOKENIZER_NAME,
        "chunks_path": str(chunks_path.resolve()),
        "chunks_size": stat.st_size,
        "chunks_mtime_ns": stat.st_mtime_ns,
    }


def cache_metadata_matches(chunks_path: Path, metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            cached_metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return cached_metadata == build_bm25_cache_metadata(chunks_path)


def build_bm25_index(chunks: Sequence[dict]) -> BM25Okapi:
    corpus_tokens = [tokenize(chunk.get("text", "")) for chunk in chunks]
    return BM25Okapi(corpus_tokens)


def load_or_build_bm25_index(
    chunks_path: Path,
    chunks: Sequence[dict],
    cache_path: Path | None = None,
    force_rebuild: bool = False,
) -> tuple[BM25Okapi, Path]:
    cache_path, metadata_path = get_bm25_cache_paths(chunks_path, cache_path)

    if not force_rebuild and cache_path.exists() and cache_metadata_matches(chunks_path, metadata_path):
        try:
            with cache_path.open("rb") as handle:
                bm25 = pickle.load(handle)
            if isinstance(bm25, BM25Okapi):
                LOG.info("Loaded cached BM25 index from %s", cache_path)
                return bm25, cache_path
            LOG.warning("Ignoring BM25 cache with unexpected type at %s", cache_path)
        except Exception as exc:
            LOG.warning("Failed to load BM25 cache from %s: %s", cache_path, exc)

    LOG.info("Building BM25 index from %s", chunks_path)
    bm25 = build_bm25_index(chunks)
    metadata = build_bm25_cache_metadata(chunks_path)
    atomic_write(
        cache_path,
        lambda tmp_path: tmp_path.write_bytes(pickle.dumps(bm25, protocol=pickle.HIGHEST_PROTOCOL)),
    )
    atomic_write(
        metadata_path,
        lambda tmp_path: tmp_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        ),
    )
    LOG.info("Saved BM25 index cache to %s", cache_path)
    return bm25, cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline BM25-over-chunks RAG runner")
    parser.add_argument("--questions-file", required=True, help="Input path with one question per line")
    parser.add_argument("--answers-file", required=True, help="Output path for one prediction per line")
    parser.add_argument(
        "--chunks-path",
        default=str(DEFAULT_CHUNKS_PATH.relative_to(SCRIPT_DIR)),
        help="Path to chunk artifact JSON, relative to repo root by default",
    )
    parser.add_argument(
        "--bm25-cache-path",
        default=None,
        help="Optional path for the serialized BM25 cache; defaults to a sidecar next to the chunk file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=ALLOWED_MODELS,
        help="Allowed OpenRouter model to use",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks to include per question",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Per-question LLM timeout in seconds",
    )
    parser.add_argument(
        "--timeout-retries",
        type=int,
        default=DEFAULT_TIMEOUT_RETRIES,
        help="Number of retries for OpenRouter timeout errors",
    )
    parser.add_argument(
        "--timeout-retry-backoff-sec",
        type=float,
        default=DEFAULT_TIMEOUT_RETRY_BACKOFF_SEC,
        help="Base backoff in seconds between timeout retries",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="Force rebuilding the BM25 cache even if a matching cached index already exists",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    chunks_path = resolve_path(args.chunks_path, default=DEFAULT_CHUNKS_PATH)
    assert chunks_path is not None
    bm25_cache_path = resolve_path(args.bm25_cache_path)
    validate_runtime_env(chunks_path)

    chunks = load_chunks(chunks_path)
    bm25, bm25_cache_path = load_or_build_bm25_index(
        chunks_path=chunks_path,
        chunks=chunks,
        cache_path=bm25_cache_path,
        force_rebuild=args.rebuild_bm25,
    )
    retriever = BM25Retriever(chunks, bm25)
    LOG.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    LOG.info("Using BM25 backend: %s", retriever.backend_name)
    LOG.info("Using BM25 cache: %s", bm25_cache_path)

    questions_path = Path(args.questions_file)
    answers_path = Path(args.answers_file)
    questions = questions_path.read_text(encoding="utf-8").splitlines()

    answers: list[str] = []
    for idx, question in enumerate(questions, start=1):
        LOG.info("Answering question %d/%d", idx, len(questions))
        answer = answer_question(
            question=question,
            retriever=retriever,
            model=args.model,
            top_k=max(1, args.top_k),
            timeout_sec=max(1, args.timeout_sec),
            timeout_retries=max(0, args.timeout_retries),
            timeout_retry_backoff_sec=max(0.0, args.timeout_retry_backoff_sec),
        )
        answers.append(answer)

    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with answers_path.open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(answer + "\n")

    LOG.info("Wrote %d predictions to %s", len(answers), answers_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
