#!/usr/bin/env python3
"""
Shared retrieval-agnostic helpers for local RAG experiments.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

from llm import ALLOWED_MODELS, call_llm

LOG = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNKS_PATH = SCRIPT_DIR / "scrape" / "corpus_full" / "chunks.json"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
#DEFAULT_MODEL = "qwen/qwen3-8b"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT_SEC = 5
DEFAULT_TIMEOUT_RETRIES = 2
DEFAULT_TIMEOUT_RETRY_BACKOFF_SEC = 1.0
DEFAULT_MAX_TOKENS = 32
DEFAULT_MODEL_TEMPERATURE = 0.0
PLACEHOLDER_ANSWER = "Unknown"
MAX_PROMPT_WORDS_PER_CHUNK = 220
MAX_WORDS_IN_ANSWER = 10

SYSTEM_PROMPT = (
    "You are answering factoid questions from retrieved EECS website snippets.\n"
    "Read all snippets before answering; the correct snippet may appear later, not first.\n"
    "Return only the final answer string.\n"
    "Do not explain your reasoning.\n"
    "Do not write a sentence, preamble, quote, markdown, or label.\n"
    "When possible, copy the exact shortest answer span from the context.\n"
    "For yes/no questions, return exactly Yes or No.\n"
    "For numbers, GPAs, years, scores, course numbers, phone numbers, dates, emails, "
    "names, and URLs, copy the exact value from the context and answer in number only.\n"
    "Do not add extra words that are not required by the answer.\n"
    "If the answer is a number, output the number and do not spell it out in words.\n"
    "Do not output any non UTF-8 characters.\n"
    "If the answer is not supported by the snippets, return Unknown.\n"
    "Examples of good outputs: 4833 | No | 3 | CS 70 | memorial@eecs.berkeley.edu | "
    "March 23, 1868 | GRASP | Zoom\n"
    "Examples of bad outputs: Based on the provided context, the answer is 4833. | "
    "Applicants should use the institution code 4833. | three main components | 3.0 GPA "
    "| 5 weeks | GRASP Lab | Zoom Webinar"
)


class Retriever(Protocol):
    backend_name: str

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        ...


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_path(path_str: str | None, default: Path | None = None) -> Path | None:
    """Resolve a repo-relative path, optionally falling back to a default path."""
    if not path_str:
        if default is None:
            return None
        path = default
    else:
        path = Path(path_str)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load the chunk artifact from disk."""
    with chunks_path.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Chunk artifact at {chunks_path} is empty or invalid")
    required = {"text", "raw_text", "title", "url"}
    missing = required - set(chunks[0].keys())
    if missing:
        raise ValueError(f"Chunk artifact at {chunks_path} is missing keys: {sorted(missing)}")
    return chunks


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        write_fn(tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def compact_passage(text: str, max_words: int = MAX_PROMPT_WORDS_PER_CHUNK) -> str:
    """Compact the passage by removing extra whitespace."""
    words = text.split()
    return " ".join(words)
    # if len(words) <= max_words:
    #     return " ".join(words)
    # return " ".join(words[:max_words]) + " ..."


def build_prompt(question: str, retrieved_chunks: Sequence[dict]) -> str:
    """Build the prompt for the LLM."""
    parts = [
        f"Question: {question}",
        "",
        "Instructions:",
        "- Read every snippet before answering.",
        "- Return one line with only the final answer.",
        "- If the snippets do not support an answer, return Unknown.",
        "",
        "Snippets:",
    ]
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.get("title", "").strip() or "(untitled)"
        url = chunk.get("url", "").strip() or "(missing url)"
        raw_text = compact_passage(chunk.get("raw_text", ""))
        parts.append(f"[{rank}] Title: {title}")
        parts.append(f"[{rank}] URL: {url}")
        parts.append(f"[{rank}] Passage: {raw_text}")
        parts.append("")
    parts.append("Final answer:")
    return "\n".join(parts)


def clean_answer(answer: str) -> str:
    """Clean markdown wrappers and enforce a short single-line answer."""
    lines = answer.splitlines()
    line = lines[0] if lines else ""
    line = line.strip()
    if line.startswith("```") and line.endswith("```") and len(line) >= 6:
        line = line.strip("`").strip()
    line = line.strip(" \t\r\n'\"")
    line = re.sub(r"\s+", " ", line)
    words = line.split()
    if not words:
        return PLACEHOLDER_ANSWER
    if len(words) > MAX_WORDS_IN_ANSWER:
        line = " ".join(words[:MAX_WORDS_IN_ANSWER])
    return line or PLACEHOLDER_ANSWER


def is_timeout_error(exc: Exception) -> bool:
    return "OpenRouter request timed out" in str(exc)


def validate_runtime_env(chunks_path: Path) -> None:
    if not chunks_path.exists():
        raise SystemExit(f"Chunk artifact not found: {chunks_path}")


def answer_question(
    question: str,
    retriever: Retriever,
    model: str,
    top_k: int,
    timeout_sec: int,
    timeout_retries: int,
    timeout_retry_backoff_sec: float,
) -> str:
    question = question.strip()
    if not question:
        return PLACEHOLDER_ANSWER

    retrieved_chunks = retriever.retrieve(question, top_k=top_k)
    if not retrieved_chunks:
        return PLACEHOLDER_ANSWER

    prompt = build_prompt(question, retrieved_chunks)
    max_attempts = max(1, timeout_retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            raw_answer = call_llm(
                query=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_MODEL_TEMPERATURE,
                timeout=timeout_sec,
            )
            return clean_answer(raw_answer)
        except Exception as exc:
            if is_timeout_error(exc) and attempt < max_attempts:
                delay_sec = timeout_retry_backoff_sec * attempt
                LOG.warning(
                    "Question timed out on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    delay_sec,
                )
                time.sleep(delay_sec)
                continue
            LOG.warning("Question failed; using placeholder answer. Reason: %s", exc)
            return PLACEHOLDER_ANSWER

    return PLACEHOLDER_ANSWER
