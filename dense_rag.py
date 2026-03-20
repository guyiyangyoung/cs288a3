#!/usr/bin/env python3
"""
Dense-retrieval RAG entrypoint.

Builds/loads a FAISS index over sentence-transformer chunk embeddings, then
retrieves top chunks for each question and answers via llm.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from query_expansion import rewrite_query_dense

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
DENSE_CACHE_VERSION = 1
DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ENCODER_BATCH_SIZE = 64


def get_dense_cache_paths(
    chunks_path: Path,
    index_path: Path | None = None,
) -> tuple[Path, Path]:
    if index_path is None:
        index_path = chunks_path.with_suffix(".dense.faiss")
    metadata_path = index_path.with_suffix(".meta.json")
    return index_path, metadata_path


def build_dense_cache_metadata(
    chunks_path: Path,
    encoder_model: str,
    normalize_embeddings: bool,
) -> dict[str, object]:
    stat = chunks_path.stat()
    return {
        "cache_version": DENSE_CACHE_VERSION,
        "chunks_path": str(chunks_path.resolve()),
        "chunks_size": stat.st_size,
        "chunks_mtime_ns": stat.st_mtime_ns,
        "encoder_model": encoder_model,
        "normalize_embeddings": normalize_embeddings,
    }


def cache_metadata_matches(
    chunks_path: Path,
    metadata_path: Path,
    encoder_model: str,
    normalize_embeddings: bool,
) -> bool:
    if not metadata_path.exists():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            cached_metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    expected = build_dense_cache_metadata(
        chunks_path=chunks_path,
        encoder_model=encoder_model,
        normalize_embeddings=normalize_embeddings,
    )
    return cached_metadata == expected


def chunk_text_for_embedding(chunk: dict) -> str:
    text = (chunk.get("text") or "").strip()
    if text:
        return text
    return (chunk.get("raw_text") or "").strip()


def build_faiss_index(
    chunks: Sequence[dict],
    encoder: SentenceTransformer,
    batch_size: int,
    normalize_embeddings: bool,
) -> faiss.Index:
    texts = [chunk_text_for_embedding(chunk) for chunk in chunks]
    embeddings = encoder.encode(
        texts,
        batch_size=max(1, batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype(np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError("Unexpected embedding shape while building dense index")

    dim = int(embeddings.shape[1])
    index: faiss.Index
    if normalize_embeddings:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def load_or_build_dense_index(
    chunks_path: Path,
    chunks: Sequence[dict],
    encoder_model: str,
    batch_size: int,
    normalize_embeddings: bool,
    index_path: Path | None = None,
    force_rebuild: bool = False,
) -> tuple[SentenceTransformer, faiss.Index, Path]:
    index_path, metadata_path = get_dense_cache_paths(chunks_path, index_path)
    encoder = SentenceTransformer(encoder_model)

    if (
        not force_rebuild
        and index_path.exists()
        and cache_metadata_matches(
            chunks_path=chunks_path,
            metadata_path=metadata_path,
            encoder_model=encoder_model,
            normalize_embeddings=normalize_embeddings,
        )
    ):
        try:
            index = faiss.read_index(str(index_path))
            if index.ntotal == len(chunks):
                LOG.info("Loaded cached dense index from %s", index_path)
                return encoder, index, index_path
            LOG.warning(
                "Ignoring dense index at %s due to ntotal mismatch (%d vs %d)",
                index_path,
                index.ntotal,
                len(chunks),
            )
        except Exception as exc:
            LOG.warning("Failed to load dense index from %s: %s", index_path, exc)

    LOG.info("Building dense FAISS index from %s", chunks_path)
    index = build_faiss_index(
        chunks=chunks,
        encoder=encoder,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    metadata = build_dense_cache_metadata(
        chunks_path=chunks_path,
        encoder_model=encoder_model,
        normalize_embeddings=normalize_embeddings,
    )
    atomic_write(
        index_path,
        lambda tmp_path: faiss.write_index(index, str(tmp_path)),
    )
    atomic_write(
        metadata_path,
        lambda tmp_path: tmp_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        ),
    )
    LOG.info("Saved dense FAISS index to %s", index_path)
    return encoder, index, index_path


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict],
        encoder: SentenceTransformer,
        index: faiss.Index,
        normalize_embeddings: bool,
    ):
        self.chunks = chunks
        self.encoder = encoder
        self.index = index
        self.normalize_embeddings = normalize_embeddings
        self.backend_name = f"faiss_dense({self.encoder.__class__.__name__})"

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        query = rewrite_query_dense(question.strip())
        if not query:
            return []

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype(np.float32)

        search_k = min(len(self.chunks), max(top_k * 6, top_k + 12))
        _, indices = self.index.search(query_embedding, search_k)
        ranked_ids = indices[0].tolist()

        results: list[dict] = []
        seen_passages: set[tuple[str, str]] = set()
        for idx in ranked_ids:
            if len(results) >= top_k:
                break
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            signature = (chunk.get("url", ""), chunk.get("raw_text", ""))
            if signature in seen_passages:
                continue
            seen_passages.add(signature)
            results.append(chunk)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense retrieval + LLM RAG runner")
    parser.add_argument("--questions-file", required=True, help="Input path with one question per line")
    parser.add_argument("--answers-file", required=True, help="Output path for one prediction per line")
    parser.add_argument(
        "--chunks-path",
        default=str(DEFAULT_CHUNKS_PATH.relative_to(SCRIPT_DIR)),
        help="Path to chunk artifact JSON, relative to repo root by default",
    )
    parser.add_argument(
        "--dense-index-path",
        default=None,
        help="Optional path for serialized FAISS index; defaults to a sidecar next to chunk file",
    )
    parser.add_argument(
        "--encoder-model",
        default=DEFAULT_ENCODER_MODEL,
        help="Sentence-transformer model name (<400M params recommended)",
    )
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=DEFAULT_ENCODER_BATCH_SIZE,
        help="Batch size used while embedding chunks for index build",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable embedding normalization (uses L2 FAISS index instead of cosine/IP)",
    )
    parser.add_argument(
        "--rebuild-dense-index",
        action="store_true",
        help="Force rebuilding dense index even if a matching cache exists",
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
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    chunks_path = resolve_path(args.chunks_path, default=DEFAULT_CHUNKS_PATH)
    assert chunks_path is not None
    dense_index_path = resolve_path(args.dense_index_path)
    normalize_embeddings = not args.no_normalize_embeddings

    validate_runtime_env(chunks_path)
    chunks = load_chunks(chunks_path)

    encoder, index, dense_index_path = load_or_build_dense_index(
        chunks_path=chunks_path,
        chunks=chunks,
        encoder_model=args.encoder_model,
        batch_size=max(1, args.encoder_batch_size),
        normalize_embeddings=normalize_embeddings,
        index_path=dense_index_path,
        force_rebuild=args.rebuild_dense_index,
    )
    retriever = DenseRetriever(
        chunks=chunks,
        encoder=encoder,
        index=index,
        normalize_embeddings=normalize_embeddings,
    )

    LOG.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    LOG.info("Using dense encoder: %s", args.encoder_model)
    LOG.info("Using dense backend: %s", retriever.backend_name)
    LOG.info("Using dense index cache: %s", dense_index_path)

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
