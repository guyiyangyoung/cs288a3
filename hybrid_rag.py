#!/usr/bin/env python3
"""
Hybrid-retrieval RAG entrypoint.

Combines BM25 and dense retrieval using weighted reciprocal-rank fusion (RRF),
then answers questions through llm.py.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from query_expansion import expand_query_bm25, rewrite_query_dense

import numpy as np

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
    configure_logging,
    load_chunks,
    resolve_path,
    validate_runtime_env,
)

LOG = logging.getLogger(__name__)
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 6
DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#DEFAULT_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"
#DEFAULT_ENCODER_MODEL = "intfloat/e5-base-v2"
DEFAULT_ENCODER_BATCH_SIZE = 64
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict],
        bm25,
        encoder,
        index,
        normalize_embeddings: bool,
        rrf_k: int,
        bm25_weight: float,
        dense_weight: float,
        candidate_multiplier: int,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self.encoder = encoder
        self.index = index
        self.normalize_embeddings = normalize_embeddings
        self.rrf_k = max(1, rrf_k)
        self.bm25_weight = max(0.0, bm25_weight)
        self.dense_weight = max(0.0, dense_weight)
        self.candidate_multiplier = max(2, candidate_multiplier)
        self.backend_name = (
            "hybrid_rrf("
            f"bm25={self.bm25_weight:.2f},dense={self.dense_weight:.2f},k={self.rrf_k}"
            ")"
        )

    def _bm25_ranked_ids(self, query_tokens: list[str], search_k: int) -> list[int]:
        # NEW: accepts pre-expanded tokens instead of raw question
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        ids: list[int] = []
        for idx, score in ranked:
            if len(ids) >= search_k:
                break
            if score <= 0:
                continue
            ids.append(idx)
        return ids

    def _dense_ranked_ids(self, dense_query: str, search_k: int) -> list[int]:
        # NEW: accepts rewritten query instead of raw question
        if not dense_query.strip():
            return []
        query_embedding = self.encoder.encode(
            [dense_query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype(np.float32)
        _, indices = self.index.search(query_embedding, search_k)
        return [idx for idx in indices[0].tolist() if 0 <= idx < len(self.chunks)]

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        query = question.strip()
        if not query:
            return []

        # NEW: expand query differently for each retriever
        bm25_tokens = expand_query_bm25(query)
        dense_query = rewrite_query_dense(query)
 
        search_k = min(len(self.chunks), max(top_k * self.candidate_multiplier, top_k + 12))
        bm25_ids = self._bm25_ranked_ids(bm25_tokens, search_k=search_k)      # NEW: pass tokens
        dense_ids = self._dense_ranked_ids(dense_query, search_k=search_k)     # NEW: pass rewritten
 
        fused_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ids, start=1):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + self.bm25_weight / (self.rrf_k + rank)
        for rank, idx in enumerate(dense_ids, start=1):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + self.dense_weight / (self.rrf_k + rank)
 
        ranked_ids = sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)
        results: list[dict] = []
        seen_passages: set[tuple[str, str]] = set()
        for idx in ranked_ids:
            if len(results) >= top_k:
                break
            chunk = self.chunks[idx]
            signature = (chunk.get("url", ""), chunk.get("raw_text", ""))
            if signature in seen_passages:
                continue
            seen_passages.add(signature)
            results.append(chunk)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid BM25 + Dense retrieval + LLM RAG runner")
    parser.add_argument("--questions-file", required=True, help="Input path with one question per line")
    parser.add_argument("--answers-file", required=True, help="Output path for one prediction per line")
    parser.add_argument(
        "--skip-existing-answers",
        action="store_true",
        help="Skip generation when answers-file already exists with matching line count",
    )
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
        "--rebuild-bm25",
        action="store_true",
        help="Force rebuilding the BM25 cache even if a matching cached index already exists",
    )
    parser.add_argument(
        "--rebuild-dense-index",
        action="store_true",
        help="Force rebuilding dense index even if a matching cache exists",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF offset constant; higher values flatten rank differences",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="Weight for BM25 ranking in RRF fusion",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Weight for dense ranking in RRF fusion",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=DEFAULT_CANDIDATE_MULTIPLIER,
        help="Retrieve this multiple of top-k before fusion",
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
    questions_path = Path(args.questions_file)
    answers_path = Path(args.answers_file)
    questions = questions_path.read_text(encoding="utf-8").splitlines()

    if args.skip_existing_answers and answers_path.exists():
        existing_answers = answers_path.read_text(encoding="utf-8").splitlines()
        if len(existing_answers) == len(questions):
            LOG.info(
                "Skipping generation for %s because it already exists with %d answers",
                answers_path,
                len(existing_answers),
            )
            return 0
        LOG.warning(
            "Existing answers file %s has %d lines but questions has %d lines; regenerating",
            answers_path,
            len(existing_answers),
            len(questions),
        )

    chunks_path = resolve_path(args.chunks_path, default=DEFAULT_CHUNKS_PATH)
    assert chunks_path is not None
    bm25_cache_path = resolve_path(args.bm25_cache_path)
    dense_index_path = resolve_path(args.dense_index_path)
    normalize_embeddings = not args.no_normalize_embeddings

    from bm25_rag import load_or_build_bm25_index
    from dense_rag import load_or_build_dense_index

    validate_runtime_env(chunks_path)
    chunks = load_chunks(chunks_path)

    bm25, bm25_cache_path = load_or_build_bm25_index(
        chunks_path=chunks_path,
        chunks=chunks,
        cache_path=bm25_cache_path,
        force_rebuild=args.rebuild_bm25,
    )
    encoder, index, dense_index_path = load_or_build_dense_index(
        chunks_path=chunks_path,
        chunks=chunks,
        encoder_model=args.encoder_model,
        batch_size=max(1, args.encoder_batch_size),
        normalize_embeddings=normalize_embeddings,
        index_path=dense_index_path,
        force_rebuild=args.rebuild_dense_index,
    )

    retriever = HybridRetriever(
        chunks=chunks,
        bm25=bm25,
        encoder=encoder,
        index=index,
        normalize_embeddings=normalize_embeddings,
        rrf_k=max(1, args.rrf_k),
        bm25_weight=max(0.0, args.bm25_weight),
        dense_weight=max(0.0, args.dense_weight),
        candidate_multiplier=max(2, args.candidate_multiplier),
    )

    LOG.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    LOG.info("Using hybrid backend: %s", retriever.backend_name)
    LOG.info("Using BM25 cache: %s", bm25_cache_path)
    LOG.info("Using dense index cache: %s", dense_index_path)

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
