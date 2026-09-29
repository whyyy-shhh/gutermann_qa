"""
CLI RAG question-answering agent for the Gutermann product catalogue.

Run:
    python qa_agent.py

The application:
1. Loads product_overview.md.
2. Splits it into product-level chunks.
3. Creates local sentence-transformer embeddings.
4. Retrieves the most relevant chunks for each query.
5. Sends the retrieved chunks to either:
   - Local Ollama
   - Google Gemini
6. Prints a grounded answer and the source product sections.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# IMPORTANT:
# Load .env BEFORE reading any environment variables.
load_dotenv(BASE_DIR / ".env")

DOCUMENT_PATH = BASE_DIR / "product_overview.md"

# Which LLM provider to use:
#   ollama
#   gemini
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama",
).lower()

# Ollama model
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2.5:3b",
)

# Ollama API
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

# Gemini model
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

# Gemini API key
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
)

# Local embedding model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

# Number of chunks retrieved
TOP_K = int(
    os.getenv(
        "TOP_K",
        "5",
    )
)

# Minimum retrieval score
MIN_SIMILARITY = float(
    os.getenv(
        "MIN_SIMILARITY",
        "0.20",
    )
)


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Chunk:
    """Represents one product section from the catalogue."""

    title: str
    category: str
    text: str
    embedding: np.ndarray | None = None


# ============================================================
# MARKDOWN CLEANING
# ============================================================

def clean_markdown(text: str) -> str:
    """Remove unnecessary Markdown/HTML elements."""

    # Remove HTML comments.
    text = re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove Markdown image syntax.
    text = re.sub(
        r"!\[[^\]]*\]\([^)]*\)",
        "",
        text,
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# CATEGORY DETECTION
# ============================================================

def infer_category(lines_before: List[str]) -> str:
    """Find the nearest previous ## heading."""

    for line in reversed(lines_before):

        match = re.match(
            r"^##\s+(.+?)\s*$",
            line,
        )

        if match:
            return match.group(1).strip()

    return "Product information"


# ============================================================
# CHUNKING
# ============================================================

def chunk_document(path: Path) -> List[Chunk]:
    """Split the catalogue into product-level chunks while preserving parent-section context."""

    raw = path.read_text(
        encoding="utf-8"
    )

    raw = clean_markdown(raw)

    lines = raw.splitlines()

    chunks: List[Chunk] = []

    current_title: str | None = None
    current_lines: List[str] = []

    current_category = "Product information"

    # Stores the general information belonging to the current ## section.
    section_context: List[str] = []

    def flush() -> None:

        nonlocal current_title
        nonlocal current_lines

        if not current_title:
            return

        body = "\n".join(
            current_lines
        ).strip()

        body = re.sub(
            r"\n{3,}",
            "\n\n",
            body,
        )

        # Include the parent section's shared context
        # together with the product-specific information.
        parent_context = "\n".join(
            section_context
        ).strip()

        if parent_context:
            text = (
                f"{current_category}\n\n"
                f"{parent_context}\n\n"
                f"{current_title}\n"
                f"{body}"
            )
        else:
            text = (
                f"{current_category}\n\n"
                f"{current_title}\n"
                f"{body}"
            )

        if body:

            chunks.append(
                Chunk(
                    title=current_title,
                    category=current_category,
                    text=text,
                )
            )

        current_title = None
        current_lines = []

    for line in lines:

        # ----------------------------------------------------
        # Category / parent section heading
        # ----------------------------------------------------

        if line.startswith("## "):

            flush()

            current_category = line[3:].strip()

            # Start collecting the general information
            # belonging to this parent section.
            section_context = []

            continue

        # ----------------------------------------------------
        # Product heading
        # ----------------------------------------------------

        product_match = re.match(
            r"^###\s+(.+?)\s*$",
            line,
        )

        if product_match:

            flush()

            current_title = (
                product_match.group(1).strip()
            )

            current_category = infer_category(
                [f"## {current_category}"]
            )

            continue

        # ----------------------------------------------------
        # Product content
        # ----------------------------------------------------

        if current_title:

            if line.strip().startswith(
                "<!-- page:"
            ):
                continue

            current_lines.append(line)

        else:

            # Lines appearing before the first product
            # belong to the parent section.
            if not line.strip().startswith(
                "<!-- page:"
            ):
                section_context.append(line)

    # Save final product.
    flush()

    if not chunks:

        raise ValueError(
            "No product sections were found "
            "in the Markdown document."
        )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def build_index(
    chunks: List[Chunk],
    model: SentenceTransformer,
) -> np.ndarray:
    """Create embeddings for all document chunks."""

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        chunk.embedding = embedding

    return embeddings


# ============================================================
# LEXICAL SIMILARITY
# ============================================================

def lexical_overlap(
    query: str,
    text: str,
) -> float:
    """Calculate overlap between query words and document words."""

    query_terms = set(
        re.findall(
            r"[a-z0-9]+",
            query.lower(),
        )
    )

    text_terms = set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )

    if not query_terms:
        return 0.0

    return len(
        query_terms & text_terms
    ) / len(query_terms)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    chunks: List[Chunk],
    matrix: np.ndarray,
    model: SentenceTransformer,
    top_k: int = TOP_K,
):
    """Retrieve the most relevant product chunks."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Since embeddings are normalized,
    # dot product = cosine similarity.
    semantic_scores = (
        matrix @ query_embedding
    )

    scored = []

    for i, chunk in enumerate(chunks):

        lexical = lexical_overlap(
            query,
            chunk.text,
        )

        # Combine semantic + lexical similarity.
        combined_score = (
            0.70 * float(
                semantic_scores[i]
            )
            + 0.30 * lexical
        )

        scored.append(
            (
                chunk,
                combined_score,
                float(
                    semantic_scores[i]
                ),
            )
        )

    # Highest score first.
    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored[:top_k]


# ============================================================
# CONTEXT CREATION
# ============================================================

def make_context(results) -> str:
    """Create the context given to the LLM."""

    blocks = []

    for rank, (
        chunk,
        combined_score,
        semantic_score,
    ) in enumerate(
        results,
        start=1,
    ):

        blocks.append(
            f"[Source {rank}]\n"
            f"Product: {chunk.title}\n"
            f"Category: {chunk.category}\n"
            f"Content:\n{chunk.text}"
        )

    return "\n\n---\n\n".join(
        blocks
    )


# ============================================================
# SHARED PROMPT
# ============================================================

def build_prompt(
    query: str,
    results,
) -> str:
    """Build the grounded prompt shared by both LLM providers."""

    context = make_context(results)

    instructions = """
You are a product knowledge assistant for a technical assessment.

Answer the user's question using ONLY the supplied document context.

Do not use outside knowledge.

IMPORTANT RULES:

1. If the context contains the answer, answer clearly and concisely.

2. For comparisons, mention only differences supported by the document.

3. For recommendation questions, recommend a product only when the
   document provides evidence that it matches the user's requirements.

4. If the document does not directly answer the exact question, but contains
   relevant information, provide that relevant information and clearly state
   what the document does not confirm. If no relevant information is available,
   explicitly say that the information is not available in the provided document.

5. Never invent pricing, availability, ordering information, specifications,
   dates, or features.

6. Do not turn an inference into a confirmed fact.

7. For availability or ordering questions, always report relevant
   availability information found in the document. Clearly distinguish
   between:
   - what the document explicitly says
   - what the document does NOT confirm.
   Do not answer "information is not available" merely because the
   document does not explicitly confirm the current availability.

8. Keep similarly named products distinct.

9. Do not mention retrieval scores unless the user asks.

10. Answer only what the user is asking for.
    Keep the answer concise, usually 1-3 sentences.
    Include a product name and only the evidence directly needed
    to answer the question.
    Do not add availability, launch dates, connectivity percentages,
    specifications, or other product details unless the user explicitly
    asks about them or they are necessary to answer the question.

"""

    return f"""
{instructions}

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""


# ============================================================
# OLLAMA GENERATION
# ============================================================

def generate_with_ollama(
    query: str,
    results,
) -> str:
    """Generate an answer using local Ollama."""

    prompt = build_prompt(
        query,
        results,
    )

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Ollama took too long to respond."
        )

    data = response.json()

    if "response" not in data:

        raise RuntimeError(
            f"Unexpected response from Ollama: {data}"
        )

    return data[
        "response"
    ].strip()


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_gemini(
    client,
    query: str,
    results,
) -> str:
    """Generate an answer using Google Gemini."""

    prompt = build_prompt(
        query,
        results,
    )

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Gemini request failed: {exc}"
        )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


# ============================================================
# LLM ROUTER
# ============================================================

def generate_answer(
    query: str,
    results,
    gemini_client=None,
) -> str:
    """Generate an answer using the selected provider."""

    if LLM_PROVIDER == "ollama":

        return generate_with_ollama(
            query,
            results,
        )

    if LLM_PROVIDER == "gemini":

        if gemini_client is None:

            raise RuntimeError(
                "Gemini client was not initialized."
            )

        return generate_with_gemini(
            gemini_client,
            query,
            results,
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
        "Use 'ollama' or 'gemini'."
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def print_sources(results) -> None:
    """Display the retrieved product sections."""

    print("\nSources used:")

    seen = set()

    if not results:
        return

    for chunk, combined_score, _ in results:
        if chunk.title in seen:
            continue

        seen.add(chunk.title)

        print(
            f"  - {chunk.title} "
            f"({chunk.category})"
        )

# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:

    # .env has already been loaded above.
    # Do NOT load it again here.

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if not DOCUMENT_PATH.exists():

        print(
            f"ERROR: Could not find "
            f"{DOCUMENT_PATH}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # INGESTION
    # --------------------------------------------------------

    print(
        "Loading product catalogue..."
    )

    chunks = chunk_document(
        DOCUMENT_PATH
    )

    print(
        f"Created {len(chunks)} "
        f"product chunks."
    )

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )

    # Create vector index.
    matrix = build_index(
        chunks,
        embedding_model,
    )

    # --------------------------------------------------------
    # INITIALIZE SELECTED LLM
    # --------------------------------------------------------

    gemini_client = None

    if LLM_PROVIDER == "ollama":

        print(
            f"Using local Ollama model: "
            f"{LLM_MODEL}"
        )

        print(
            f"Ollama endpoint: "
            f"{OLLAMA_URL}"
        )

    elif LLM_PROVIDER == "gemini":

        if not GEMINI_API_KEY:

            print(
                "ERROR: GEMINI_API_KEY is not set."
            )

            print(
                "Add your Gemini API key to "
                ".env as GEMINI_API_KEY=..."
            )

            sys.exit(1)

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            f"Using Gemini model: "
            f"{GEMINI_MODEL}"
        )

    else:

        print(
            f"ERROR: Unknown LLM_PROVIDER: "
            f"{LLM_PROVIDER}"
        )

        print(
            "Use LLM_PROVIDER=ollama "
            "or LLM_PROVIDER=gemini"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # CLI
    # --------------------------------------------------------

    print(
        "\nGutermann Product QA Agent"
    )

    print(
        "Type a question, or type "
        "'exit' / 'quit' to stop."
    )

    while True:

        try:

            query = input(
                "\n> "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nGoodbye!"
            )

            break

        # Ignore empty input.
        if not query:
            continue

        # Exit.
        if query.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye!"
            )

            break

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        results = retrieve(
            query,
            chunks,
            matrix,
            embedding_model,
        )

        # ----------------------------------------------------
        # RELEVANCE CHECK
        # ----------------------------------------------------

        if (
            not results
            or results[0][1] < MIN_SIMILARITY
        ):

            print(
                "\nAgent: The provided document "
                "does not contain enough "
                "information to answer that "
                "question."
            )

            continue

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        try:

            answer = generate_answer(
                query,
                results,
                gemini_client,
            )

            print(
                f"\nAgent: {answer}"
            )

            print_sources(
                results
            )

        except Exception as exc:

            print(
                "\nERROR while generating "
                f"the answer: {exc}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()