# Smart PDF Agent

An agentic RAG PDF reader built with LangGraph and Groq. Load two or three PDFs, then ask questions across them in plain English. Answers cite the file and page they came from, and arithmetic between documents is done by a calculator tool instead of the LLM guessing.

## How it works

```
START -> route -> retrieve -> synthesize -> END
```

| Step | What it does |
|------|--------------|
| **route** | The LLM decides which of the loaded PDFs a question actually needs. |
| **retrieve** | Pulls the top chunks from each selected PDF using FAISS and `all-MiniLM-L6-v2` embeddings. Handles two-column pages and table rows separately. |
| **synthesize** | Answers using only the retrieved chunks, citing `[file, p.N]`. Calls a `calculate` tool for any arithmetic. |
| **memory** | The last few Q&A turns are carried into each new question, so follow-ups work. |

## Setup

Requires Python 3.10+.

```
pip install -r requirements.txt
```

### Getting an API key

The agent uses Groq's free tier. Nothing secret is stored in this repo, so you need your own key:

1. Create a free key at https://console.groq.com (no card needed).
2. Set it as an environment variable.

PowerShell:
```
$env:GROQ_API_KEY="your_key_here"
```

Command Prompt:
```
set GROQ_API_KEY=your_key_here
```

macOS / Linux:
```
export GROQ_API_KEY="your_key_here"
```

## Usage

PDFs are not included in the repo. Put your own PDFs in the project folder (for example the [Transformer](https://arxiv.org/abs/1706.03762) and [BERT](https://arxiv.org/abs/1810.04805) papers from arXiv), then run:

```
py rag_agent.py transformer.pdf bert.pdf
```

Ask questions at the `>` prompt. Type `exit` to quit.

### Example

```
> How many more parameters does BERT_LARGE have than the Transformer big model?

[used: transformer.pdf, bert.pdf]
BERT LARGE has 340M parameters [bert.pdf, p.3], while the Transformer big
model has 213M parameters [transformer.pdf, p.9].
The difference is: 340M - 213M = 127M parameters.
```

## Configuration

Settings at the top of `rag_agent.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EMBED_MODEL`, `GROQ_MODEL` (default `openai/gpt-oss-20b`), and `MAX_HISTORY_TURNS`.

## Known limitations

- Table extraction from PDFs is heuristic; complex tables can produce noisy chunks.
- Scanned (image-only) PDFs are not supported, since there is no OCR step.
- Answers depend on the retrieved chunks, so very vague questions may return "Not found in the documents."

## Tech stack

Python, LangGraph, Groq API, FAISS, sentence-transformers, pdfplumber.
