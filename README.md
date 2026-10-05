# Smart PDF Agent

A RAG-based PDF knowledge agent built with LangGraph and Groq. Load up to a few PDFs and ask questions across them, including questions that need arithmetic between documents.

## How it works

`START -> route -> retrieve -> synthesize -> END`

- **route**: the LLM picks which PDFs a question needs.
- **retrieve**: top-k chunks per selected PDF from a FAISS index (MiniLM embeddings). Two-column pages and table rows are handled separately.
- **synthesize**: answers only from retrieved chunks, with file and page citations. Uses a `calculate` tool for arithmetic.
- **memory**: the last few Q&A turns are carried into each new question.

## Setup

```
pip install -r requirements.txt
set GROQ_API_KEY=your_key_here      # Windows (PowerShell: $env:GROQ_API_KEY="your_key_here")
```

## Usage

```
py rag_agent.py transformer.pdf bert.pdf
> How many more parameters does BERT_LARGE have than the Transformer big model?
```

Type `exit` to quit.

## Notes

- Model: `openai/gpt-oss-20b` on Groq.
- PDFs are not committed; add your own.
