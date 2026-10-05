
"""
Step 3: LangGraph agent layer — routing, synthesis, calculator tool, chat memory.

Adds an agent graph on top of the step 2 retrieval pipeline:

    START -> route -> retrieve -> synthesize -> END

  - route:      the LLM decides which loaded PDF(s) a question actually needs
                (skips irrelevant PDFs instead of always searching everything)
  - retrieve:   pulls top-k chunks, restricted to the PDFs the router picked
  - synthesize: answers using ONLY those chunks; can call a `calculate` tool
                for arithmetic instead of doing math in its head (this is
                what fixes the "can it do arithmetic between two PDFs?"
                reliability problem from step 2 — real computation instead
                of the LLM guessing at math in free text)
  - chat memory: the last few turns of conversation are carried into each new
                question, so follow-ups like "and how does that compare to
                what we discussed earlier" work

Usage:
    python rag_agent.py file1.pdf file2.pdf [file3.pdf ...]
    (then ask questions interactively; type 'exit' to quit)

Setup (new package this step: langgraph):
    pip install pdfplumber sentence-transformers faiss-cpu groq langgraph
    set GROQ_API_KEY as an environment variable
"""

import os
import re
import sys
import json
import ast
import operator
from dataclasses import dataclass
from typing import TypedDict

import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from langgraph.graph import StateGraph, START, END

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 10  # this is now PER SOURCE (see search_per_source) — with 2 PDFs
            # selected that's 20 chunks total
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"   # current Groq free-tier model (Sept 2026)
MAX_HISTORY_TURNS = 4               # how many past Q&A pairs to keep in memory


# ---------------------------------------------------------------------------
# PDF loading + chunking (same as step 2)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = extract_page_text(page)
            text = re.sub(r"[ \t]{2,}", " ", text) 
            if text.strip():
                pages.append((i, text))
    return pages


def extract_page_text(page) -> str:
    words = page.extract_words()
    if not words:
        return page.extract_text() or ""

    width = page.width
    mid = width / 2

    # how many words straddle the vertical midline?
    # (a real two-column gutter will have ~0; a full-width table will have many)
    straddlers = sum(1 for w in words if w["x0"] < mid < w["x1"])
    crossing_ratio = straddlers / len(words)

    if crossing_ratio > 0.03:
        # looks like single-column content (table, figure, title block) --
        # don't split, just extract normally
        return page.extract_text(x_tolerance=1.5) or ""

    # treat as two columns: crop left half, then right half, concatenate
    left = page.crop((0, 0, mid, page.height))
    right = page.crop((mid, 0, width, page.height))
    left_text = left.extract_text(x_tolerance=1.5) or ""
    right_text = right.extract_text(x_tolerance=1.5) or ""

    return left_text + "\n" + right_text

def chunk_pages(pages: list[tuple[int, str]], source: str, start_id: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    cid = start_id
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, source=source, page=page_num, chunk_id=cid))
                cid += 1
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks


TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
}


def extract_table_chunks(pdf_path: str, source: str, start_id: int) -> list[Chunk]:
    """Extract each detected table row as its OWN atomic chunk, bypassing
    CHUNK_SIZE splitting entirely. This keeps a row's label (e.g.
    'BERT_LARGE') paired with its numbers (e.g. '79.6') in the same chunk."""
    chunks: list[Chunk] = []
    cid = start_id
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables(table_settings=TABLE_SETTINGS)
            for table in tables:
                for row in table:
                    cells = [c.strip() for c in row if c and c.strip()]
                    if len(cells) < 3 or not any(ch.isdigit() for ch in " ".join(cells)):
                        continue
                    row_text = " | ".join(cells)
                    chunks.append(Chunk(text=row_text, source=source, page=page_num, chunk_id=cid))
                    cid += 1
    return chunks


def load_pdfs(pdf_paths: list[str]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in pdf_paths:
        source = os.path.basename(path)
        print(f"Reading {source}...")
        pages = extract_pages(path)
        print(f"  {len(pages)} pages with text")
        chunks = chunk_pages(pages, source=source, start_id=len(all_chunks))
        print(f"  {len(chunks)} prose chunks")
        all_chunks.extend(chunks)

        table_chunks = extract_table_chunks(path, source=source, start_id=len(all_chunks))
        print(f"  {len(table_chunks)} table-row chunks")
        all_chunks.extend(table_chunks)
    return all_chunks


class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, k: int = TOP_K, allowed_sources: set[str] | None = None) -> list[Chunk]:
        """Search, optionally restricted to a subset of source filenames."""
        assert self.index is not None, "call build() first"
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        # over-fetch then filter, since FAISS doesn't filter natively here
        fetch_k = k * 4 if allowed_sources else k
        scores, idxs = self.index.search(q_emb, min(fetch_k, len(self.chunks)))
        results = [self.chunks[i] for i in idxs[0] if i != -1]
        if allowed_sources:
            results = [c for c in results if c.source in allowed_sources]
        return results[:k]

    def search_per_source(self, query: str, sources: list[str], k_per_source: int = TOP_K) -> list[Chunk]:
        """Search each source independently and concatenate results, so a
        multi-PDF question gives every selected PDF its own top-k slots
        instead of all PDFs competing for one shared top-k (which can starve
        out a fact that's real but a slightly weaker embedding match)."""
        assert self.index is not None, "call build() first"
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idxs = self.index.search(q_emb, len(self.chunks))
        ranked = [self.chunks[i] for i in idxs[0] if i != -1]

        results: list[Chunk] = []
        for source in sources:
            matches = [c for c in ranked if c.source == source][:k_per_source]
            results.extend(matches)
        return results


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.source}, p.{c.page}]\n{c.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Calculator tool — real arithmetic instead of LLM mental math
# ---------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}

def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")

def calculate(expression: str) -> str:
    expr = (expression.replace("−", "-").replace("–", "-").replace("×", "*")
            .replace("÷", "/").replace("^", "**").replace(",", ""))
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    try:
        return str(round(_eval(ast.parse(expr, mode="eval")), 6))
    except Exception as e:
        return f"Error: {e}. Retry using plain numbers and + - * / ( ) only."


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression (add, subtract, multiply, divide). Use this for any numeric computation instead of computing it yourself.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A basic arithmetic expression, e.g. '93.2 - 41.8'",
                }
            },
            "required": ["expression"],
        },
    },
}


# ---------------------------------------------------------------------------
# LangGraph state + nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    history: list[tuple[str, str]]   # (question, answer) pairs
    relevant_sources: list[str]
    retrieved: list[Chunk]
    answer: str


def make_route_node(client: Groq, all_sources: list[str]):
    def route(state: AgentState) -> AgentState:
        sources_list = ", ".join(all_sources)
        prompt = f"""Available documents: {sources_list}

Question: {state['question']}

Which document(s) are needed to answer this question? Reply with ONLY a
JSON list of filenames from the available documents, e.g. ["a.pdf", "b.pdf"].
If the question needs all documents (e.g. to compare them), include all of
them. If unsure, include all documents rather than guessing narrowly."""
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
            reasoning_effort="low",
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            picked = json.loads(match.group(0)) if match else all_sources
            picked = [s for s in picked if s in all_sources]
            if not picked:
                picked = all_sources
        except Exception:
            picked = all_sources  # fall back to searching everything
        state["relevant_sources"] = picked
        return state
    return route



def make_retrieve_node(client: Groq, store: VectorStore):
    def retrieve(state: AgentState) -> AgentState:
        state["retrieved"] = store.search_per_source(
            state["question"], state["relevant_sources"], k_per_source=TOP_K
        )
        return state
    return retrieve


SYNTH_SYSTEM_PROMPT = """You answer questions using ONLY the provided context chunks
and prior conversation turns. Each chunk is labeled [filename.pdf, p.N]; cite the
file AND page for every value you use.

Questions may require arithmetic on values found in the documents (differences,
sums, ratios, percentages), within one PDF or across PDFs. This is allowed and
expected. The final result does NOT need to appear in the text, only the inputs.
Procedure:
1. Find each input number in the context and note its source [file, p.N].
2. Call the `calculate` tool for the arithmetic. Never compute it yourself.
3. Answer with the result, showing the inputs with their citations.
Only say "Not found in the documents." if an INPUT number is missing from the
context, and name which input is missing. Never refuse just because the final
result is not written in the text.
Write plain text only: no LaTeX, no markdown math. Example: 340M - 213M = 127M.
Do not use outside knowledge. Keep answers concise."""


def make_synthesize_node(client: Groq):
    def synthesize(state: AgentState) -> AgentState:
        context = build_context(state["retrieved"])

        history_text = ""
        for q, a in state["history"][-MAX_HISTORY_TURNS:]:
            history_text += f"Previous Q: {q}\nPrevious A: {a}\n\n"

        user_content = f"{history_text}Context:\n{context}\n\nQuestion: {state['question']}"

        messages = [
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        content = None
        for _ in range(5):
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=[CALCULATOR_TOOL],
                temperature=0.1,
                max_tokens=1500,
                reasoning_effort="medium",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                content = msg.content
                break
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": calculate(args.get("expression", "")),
                })

        state["answer"] = content or "(No answer returned — try rephrasing the question.)"
        return state
    return synthesize


def build_graph(client: Groq, store: VectorStore, all_sources: list[str]):
    graph = StateGraph(AgentState)
    graph.add_node("route", make_route_node(client, all_sources))
    graph.add_node("retrieve", make_retrieve_node(client, store))
    graph.add_node("synthesize", make_synthesize_node(client))
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_agent.py file1.pdf file2.pdf [file3.pdf ...]")
        sys.exit(1)

    pdf_paths = sys.argv[1:]
    for p in pdf_paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY as an environment variable first.")
        sys.exit(1)

    print(f"Loading {len(pdf_paths)} PDF(s)...")
    chunks = load_pdfs(pdf_paths)
    all_sources = [os.path.basename(p) for p in pdf_paths]

    print("Embedding + indexing...")
    store = VectorStore()
    store.build(chunks)

    client = Groq(api_key=api_key)
    app = build_graph(client, store, all_sources)

    history: list[tuple[str, str]] = []

    print("\nReady. Ask questions across all loaded PDFs (type 'exit' to quit).\n")
    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = app.invoke({
            "question": question,
            "history": history,
            "relevant_sources": [],
            "retrieved": [],
            "answer": "",
        })


        print("\n--- retrieved ---")
        for c in result["retrieved"]:
            print(f"[{c.source} p.{c.page}] {c.text[:120]!r}")

        answer = result["answer"]
        print(f"\n[used: {', '.join(result['relevant_sources'])}]")
        print(f"{answer}\n")

        history.append((question, answer))


if __name__ == "__main__":
    main()