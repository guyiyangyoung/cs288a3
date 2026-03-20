#!/usr/bin/env python3
"""
Query Expansion for EECS RAG
=============================
Expands and rewrites queries to improve retrieval recall.
Designed to be zero-cost at runtime (dict lookups only).

Features:
  - EECS-specific abbreviation/synonym expansion
  - Query term augmentation for BM25
  - Lightweight query rewriting for dense retrieval

Usage:
    from query_expansion import expand_query_bm25, rewrite_query_dense
    
    expanded_tokens = expand_query_bm25("How many GSI hours for PhD?")
    rewritten = rewrite_query_dense("What is Dan Klein's office number?")
"""

import re
from typing import Sequence

# ---------------------------------------------------------------------------
# EECS-specific synonym / abbreviation map
# Keys are lowercased terms found in questions.
# Values are lists of expansion terms to add to the BM25 query.
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, list[str]] = {
    # Degree abbreviations
    "phd": ["doctoral", "doctorate", "ph.d.", "ph.d"],
    "ph.d.": ["phd", "doctoral", "doctorate"],
    "ph.d": ["phd", "doctoral", "doctorate"],
    "ms": ["master", "masters", "m.s."],
    "m.s.": ["ms", "master", "masters"],
    "bs": ["bachelor", "bachelors", "b.s."],
    "b.s.": ["bs", "bachelor", "bachelors"],
    "meng": ["master", "engineering", "m.eng"],
    "m.eng": ["meng", "master", "engineering"],

    # EECS-specific roles and terms
    "gsi": ["graduate", "student", "instructor", "teaching"],
    "ugsi": ["undergraduate", "student", "instructor", "teaching"],
    "gsr": ["graduate", "student", "researcher", "research"],
    "ta": ["teaching", "assistant"],
    "ra": ["research", "assistant"],
    "pi": ["principal", "investigator"],
    "faculty": ["professor", "lecturer", "instructor"],
    "professor": ["faculty", "prof"],
    "prof": ["professor", "faculty"],
    "prof.": ["professor", "faculty"],
    "lecturer": ["faculty", "instructor"],
    "postdoc": ["postdoctoral", "researcher"],
    "dean": ["associate", "dean"],
    "chair": ["department", "chair", "head"],
    "adviser": ["advisor"],
    "advisor": ["adviser"],
    "grad": ["graduate"],
    "undergrad": ["undergraduate"],

    # UC Berkeley specific
    "eecs": ["electrical", "engineering", "computer", "science", "sciences"],
    "cs": ["computer", "science"],
    "ee": ["electrical", "engineering"],
    "l&s": ["letters", "science", "college"],
    "coe": ["college", "engineering"],
    "cal": ["berkeley", "california"],
    "ucb": ["berkeley", "uc", "university", "california"],
    "uc": ["university", "california"],
    "berkeley": ["cal", "ucb"],
    "soda": ["soda", "hall"],
    "cory": ["cory", "hall"],
    "hearst": ["hearst"],
    "sdh": ["sutardja", "dai", "hall"],

    # Course-related
    "prereq": ["prerequisite", "prerequisites", "required"],
    "prerequisite": ["prereq", "required"],
    "elective": ["electives", "optional"],
    "units": ["credits", "hours"],
    "credits": ["units", "hours"],
    "gpa": ["grade", "point", "average"],
    "enrollment": ["enroll", "enrolled", "register"],
    "waitlist": ["wait", "list", "enrollment"],
    "office": ["room", "number", "location"],
    "office hours": ["oh", "hours", "availability"],
    "oh": ["office", "hours"],
    "homework": ["hw", "assignment"],
    "hw": ["homework", "assignment"],
    "midterm": ["exam", "midterm"],
    "final": ["exam", "final"],
    "lab": ["laboratory"],
    "lecture": ["class", "section"],
    "section": ["discussion", "lab"],
    "semester": ["term", "spring", "fall"],

    # Research areas
    "ai": ["artificial", "intelligence"],
    "ml": ["machine", "learning"],
    "nlp": ["natural", "language", "processing"],
    "cv": ["computer", "vision"],
    "hci": ["human", "computer", "interaction"],
    "os": ["operating", "systems"],
    "db": ["database", "databases"],
    "pl": ["programming", "languages"],
    "se": ["software", "engineering"],
    "vlsi": ["integrated", "circuits", "design"],
    "fpga": ["field", "programmable"],
    "robotics": ["robots", "robotic"],
    "cybersecurity": ["security", "cyber"],
    "iot": ["internet", "things"],

    # Administrative
    "commencement": ["graduation", "ceremony"],
    "graduation": ["commencement", "graduate", "degree"],
    "dissertation": ["thesis"],
    "thesis": ["dissertation"],
    "qualifying": ["qual", "quals", "exam"],
    "qual": ["qualifying", "exam"],
    "quals": ["qualifying", "exam", "exams"],
    "petition": ["appeal", "request"],
    "deadline": ["due", "date"],
    "due": ["deadline", "date"],
    "email": ["e-mail", "mail", "address", "contact"],
    "phone": ["telephone", "number", "contact"],
    "fax": ["facsimile", "number"],
    "website": ["web", "site", "page", "url"],
    "url": ["website", "link", "address"],
    "award": ["prize", "honor", "fellowship"],
    "prize": ["award", "honor"],
    "fellowship": ["award", "scholarship", "grant"],
    "scholarship": ["fellowship", "award", "grant"],
}

# Compile a pattern to find multi-word keys
MULTI_WORD_KEYS = {k for k in SYNONYM_MAP if " " in k}


# ---------------------------------------------------------------------------
# Query expansion for BM25
# ---------------------------------------------------------------------------

def expand_query_bm25(
    question: str,
    synonym_map: dict[str, list[str]] = SYNONYM_MAP,
    max_expansions_per_term: int = 3,
) -> list[str]:
    """
    Expand a question into an augmented list of BM25 query tokens.
    
    The original tokens come first (preserving BM25 term frequency weighting),
    followed by expansion terms. Expansion terms are added at most once each.
    
    Args:
        question: The raw question string
        synonym_map: Abbreviation/synonym mapping
        max_expansions_per_term: Cap expansions per matched term
    
    Returns:
        List of tokens for BM25 scoring
    """
    question_lower = question.lower()
    original_tokens = re.findall(r"\b\w+\b", question_lower)

    if not original_tokens:
        return original_tokens

    expansion_tokens = []
    seen_expansions = set(original_tokens)

    # Check multi-word keys first (e.g., "office hours")
    for mw_key in MULTI_WORD_KEYS:
        if mw_key in question_lower:
            expansions = synonym_map.get(mw_key, [])
            for exp in expansions[:max_expansions_per_term]:
                for tok in re.findall(r"\b\w+\b", exp.lower()):
                    if tok not in seen_expansions:
                        expansion_tokens.append(tok)
                        seen_expansions.add(tok)

    # Then check single-word keys
    for token in original_tokens:
        expansions = synonym_map.get(token, [])
        for exp in expansions[:max_expansions_per_term]:
            for tok in re.findall(r"\b\w+\b", exp.lower()):
                if tok not in seen_expansions:
                    expansion_tokens.append(tok)
                    seen_expansions.add(tok)

    return original_tokens + expansion_tokens


def rewrite_query_dense(question: str) -> str:
    """
    Lightly rewrite a question for dense retrieval.
    
    Dense models handle semantic similarity, so we don't need heavy expansion.
    Instead we:
      1. Expand key abbreviations inline for better embedding
      2. Keep the question natural-sounding
    
    Args:
        question: The raw question string
    
    Returns:
        Rewritten question string
    """
    result = question

    # Inline expansions for abbreviations that dense models struggle with
    INLINE_EXPANSIONS = {
        r"\bGSI\b": "GSI (Graduate Student Instructor)",
        r"\bUGSI\b": "UGSI (Undergraduate Student Instructor)",
        r"\bGSR\b": "GSR (Graduate Student Researcher)",
        r"\bEECS\b": "EECS (Electrical Engineering and Computer Science)",
        r"\bCS\b": "CS (Computer Science)",
        r"\bEE\b": "EE (Electrical Engineering)",
        r"\bPh\.D\.?\b": "PhD (doctoral)",
        r"\bph\.d\.?\b": "PhD (doctoral)",
        r"\bGPA\b": "GPA (grade point average)",
        r"\bPI\b": "PI (Principal Investigator)",
        r"\bSDH\b": "SDH (Sutardja Dai Hall)",
        r"\bL&S\b": "L&S (Letters and Science)",
        r"\bCoE\b": "CoE (College of Engineering)",
        r"\bNLP\b": "NLP (Natural Language Processing)",
        r"\bAI\b": "AI (Artificial Intelligence)",
        r"\bML\b": "ML (Machine Learning)",
        r"\bHCI\b": "HCI (Human-Computer Interaction)",
    }

    for pattern, replacement in INLINE_EXPANSIONS.items():
        # Only replace the first occurrence to keep query readable
        result = re.sub(pattern, replacement, result, count=1)

    return result


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def expand_for_hybrid(
    question: str,
) -> tuple[list[str], str]:
    """
    Convenience function for hybrid retrieval.
    Returns (bm25_tokens, dense_query) in one call.
    """
    bm25_tokens = expand_query_bm25(question)
    dense_query = rewrite_query_dense(question)
    return bm25_tokens, dense_query


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main():
    import sys

    test_questions = [
        "How many GSI hours do Berkeley EECS students need for a doctoral degree?",
        "What is the office number of Dan Klein?",
        "Which email address should CS PhD students send their Ph.D. Student Review to?",
        "Who is the winner of the Eugene L. Lawler Prize in 2024-25?",
        "What is the title of the dissertation of Dan Klein's most recent Ph.D. graduate?",
        "What are the prereqs for CS 61A?",
        "What is the GPA requirement for the EECS major?",
        "Where is Soda Hall located?",
        "Who is the current chair of the EECS department?",
        "What AI research labs are in EECS?",
    ]

    if len(sys.argv) > 1:
        test_questions = [" ".join(sys.argv[1:])]

    for q in test_questions:
        bm25_tokens, dense_query = expand_for_hybrid(q)

        print(f"\nQuestion: {q}")
        print(f"  BM25 tokens ({len(bm25_tokens)}): {' '.join(bm25_tokens)}")
        print(f"  Dense query: {dense_query}")
        print()


if __name__ == "__main__":
    main()