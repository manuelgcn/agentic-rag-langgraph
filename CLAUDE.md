# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

University assignment (ORT Uruguay, Taller de Tecnologías 2, obligatorio septiembre 2026), finished and delivered on 16/11/2026. The requirements are summarized under "What must be built" below.

## Setup / run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
Copy `.env.example` → `.env` (needs `CLAUDE_API_KEY`, or `GROQ_API_KEY` for Groq; Kaggle creds for `kagglehub`). LLM: `claude-sonnet-5-5` via `langchain-anthropic` (adaptive thinking at `effort="low"`, `drop_block` because `model_messages` edits history), used for the final run because Groq's daily quota ran out; developed on `openai/gpt-oss-120b` on Groq via its OpenAI-compatible endpoint (select it with `LLM_MODEL`) (free tier has rate limits). Embeddings: local `BAAI/bge-m3` (chosen over multilingual-e5 for much better relevance separation). Course PDFs go in `data/papers/`. Everything runs from `agentic_rag.ipynb` using the `.venv` kernel.

## Architecture (all in `agentic_rag.ipynb`)

Notebook sections: 1 Configuración (1.1 LLM, 1.2 embeddings y Chroma) · 2 papers · 3 dataset · 4 tools (4.1 papers → `PAPER_TOOLS`, 4.2 dataset → `DATASET_TOOLS`, 4.3 memoria → `TOOLS`, `llm_with_tools`) · 5 agente (5.1 prompt, 5.2 nodos, 5.3 memoria, 5.4 grafo) · 6 chat · 7 pruebas (one cell per test, leading comment = expected behavior) · 8 comparaciones · 9 Gradio (see below). Each section has a short markdown intro and imports what it uses. Repo layout is described in `README.md`.

- Hand-built `StateGraph(MessagesState)`: `agent` → (conditional on `tool_calls`) → `tools` → `agent`. The `tools` node is a manual loop over `tool_calls`, not `ToolNode`.
- Tool-calling robustness (5.2, the team's defense topic): `run_tool` never raises — unknown tool, args failing the schema, or an exception inside the tool come back to the LLM as an error `ToolMessage`. Malformed-JSON calls land in `invalid_tool_calls` (which `ChatOpenAI` still re-sends as tool calls), so `route`, `tools`, `tool_rounds` and `close_pending_calls` handle them too. Loop control has two caps: `MAX_TOOL_ROUNDS` (4 per user turn; then `agent` calls the unbound `llm` plus `STOP_TOOLS` to force a final answer) and LangGraph's `recursion_limit` (`RECURSION_LIMIT` = 12, caught as `GraphRecursionError` in `chat()`). `close_pending_calls` adds a `ToolMessage` for tool calls left unanswered by an interrupted turn, only in what is sent to the LLM. Section 7.3 demos all of it; the loop demos swap the global `llm_with_tools` for a stub that always asks for the same tool.
- Test battery (7.1): `prueba(nombre, message, conv=None, user_id=None)` wraps `chat()`. Without `user_id` each execution gets a fresh user and conversation (`<nombre>-<uuid>`), so neither history nor long-term memory leaks between tests or from earlier runs stored in SQLite; with `user_id` (the four memory tests) the user is `<user_id>-<RUN>`, shared within one run. It sums the turn's tokens with `get_usage_metadata_callback()` into `TOKENS`, and sleeps `tokens / TPM_LIMIT * 60` s (Groq free tier: 8K TPM) so back-to-back cells don't hit the rate limit. 22 cells (the 4 original ones plus 18 covering routing, papers, dataset, mixed, false premise, typos, fabrication requests, out-of-domain, the first GPT paper and Subliminal Learning) and a token summary cell.
- Gradio UI (9): a `gr.Blocks` app over `chat()` with user/conversation fields, the stored history reloaded from the checkpointer (`graph.get_state`) on every change, a "Nueva conversación" button (first free `conv-N` of that user) and a trace panel built by capturing `chat()`'s stdout (safe for one turn at a time). It is built on every run but only launched with `ABRIR_INTERFAZ = True`, so Run All starts no server and spends no tokens; locally it opens inline, on Colab Gradio requires a public share link. Its committed output comes from a separate run of sections 1–6 + 9 with the flag on, after the final run; the committed source keeps `False`.
- Short-term memory: `SqliteSaver` keyed by `thread_id`. Long-term memory: `SqliteStore` namespace `("memories", user_id)`; the `save_memory` tool writes to it (via `get_store()`/`get_config()`), and the `agent` node injects stored memories into the system prompt. Separate SQLite files/connections (`storage/checkpoints.sqlite`, `storage/memory.sqlite`); the store connection must use `isolation_level=None` (it issues its own BEGIN/COMMIT).
- RAG: persistent Chroma in `storage/chroma` with two collections, `papers` and `dataset`, both in cosine space (relevance score = cosine similarity). `index_documents(db, docs, ids)` in 1.2 ingests in batches with progress (shared by both ingestions). Paper ingestion cleans each page first (`clean_page`: drops repeated headers/footers and DDIA's alphabetical index; chunks < 200 chars are dropped) and only adds the chunk ids that are missing (an interrupted ingestion resumes; checking by `source` left a half-indexed PDF as done) — delete `storage/` to re-index after changing chunking/embeddings/space.
- Comparisons (8.1): `evaluate_retrieval(db, k, mmr, verbose)` gives hit@k/MRR over `RETRIEVAL_SET`; `variant_db(name, emb, chunk_size)` re-indexes all papers into an in-memory Chroma collection (never touches `storage/`); `cached(key, compute)` stores each experiment in `data/comparaciones.json` (versioned, shared with 8.2) and skips recomputing — delete the key to re-run. Results: bge-m3 and multilingual-e5-large tie on hit@5/MRR but e5 lets 7/12 off-topic queries above the worst relevant one (bge-m3: 0/12); MMR hurts; chunk 1500 ties 1000 on hit@5 with better MRR (kept 1000).
- This PC has an RTX 4070 and the `.venv` uses `torch==2.15.0+cu132` (installed with `--no-deps --index-url https://download.pytorch.org/whl/cu132`); teammates may be on CPU. `pip freeze` will print the `+cu132` suffix: strip it in `requirements.txt`.
- Paper tools (4.1): `search_papers(query, source=None, k=4)`, `list_documents()`, `paper_overview(source)`; `source` is fuzzy-resolved (`_resolve_source`: substring, acronym like "DDIA", difflib). No result above `RELEVANCE_THRESHOLD` (0.45, calibrated in 8.1 with `calibrate_threshold`) → exactly `SIN RESULTADOS RELEVANTES`; `paper_overview` skips the threshold (its generic queries score 0.43–0.56).
- PDFs are read with `pypdf` directly (one `Document` per page, metadata `source` = filename, `page` 1-based → citations like `[Attention is all you need.pdf p.3]`), so PDF filenames should be readable titles. `langchain-community` was dropped on purpose (sunset, warns on import).
- Every call needs `config={"configurable": {"thread_id": ..., "user_id": ...}}`. `chat(message, conv, user_id)` builds `thread_id = f"{user_id}:{conv}"` (and rejects a `user_id` containing `:`, which would make two pairs collide), prints the turn's trace (tool calls, args, cited sources) and returns the answer; test cells end in `;` so Jupyter does not echo it.
- Kaggle dataset: `miguelcorraljr/ted-ultimate-dataset`, only the Spanish file `2020-05-01/ted_talks_es.csv` is downloaded (`kagglehub.dataset_download(DATASET, path=CSV_FILE)`; the full dataset is ~570 MB). 3,921 TED talks with full transcripts, CC BY-NC 4.0 (topics/occupations are in English, titles/descriptions/transcripts in Spanish). Section 3 builds `TED_DF` (clean table: `speaker`, `topics` list, `duration_min`, `year` = recorded, `published_year`, …) and indexes into `dataset` one **ficha** per talk plus, only if `INDEX_TRANSCRIPTS` (default: `torch.cuda.is_available()`; env var `INDEX_TRANSCRIPTS=1`/`0` overrides), transcript chunks (1000/150, prefixed with title, speaker and year), told apart by metadata `tipo` (`ficha` / `transcripcion`). On a Colab T4 with fp16 the ~52k docs take ~12 min; on CPU only fichas are indexed. Ingestion is per missing id (`index_missing`), so an interrupted run resumes; since `index_documents` adds batches in order and stops at the first error, if the last talk's last fragment id is present the transcripts are not re-split (~48k chunks, ~6 s) nor their ids fetched; `TED_DF.speakers` lists every speaker of a talk and `duration_min` is exact (rounded only for display).
- Dataset tools (4.2): `search_dataset(query, k=3, speaker=None)` runs one search per `tipo` with its own threshold (`FICHA_THRESHOLD` 0.57, `FRAGMENTO_THRESHOLD` 0.60, calibrated on Colab T4/fp16; the cell "Umbral del dataset TED" in 8.1 repeats the measurement with `calibrate_threshold(..., where={"tipo": ...})`), at most 2 fragments per talk, `speaker` fuzzy-resolved against the main speaker only (ambiguous → candidate list; with a speaker, fichas skip the threshold). `query_dataset(filters, sort_by, descending, limit, count)` filters/sorts/counts `TED_DF` with pandas (no `eval`); topics are in English and `contains` matches whole words/phrases inside each topic ("AI" must not match "brain"), an unknown topic returns close matches, the `speaker` filter includes co-speakers, and `count=True` with no match answers "0 charlas". Both cite `[TED: título, orador (año)]` via `_cite` (4.1) and return `SIN RESULTADOS RELEVANTES` when nothing matches.

## What must be built

An **agentic RAG** system in Python, orchestrated with **LangGraph**, delivered preferably as a `.ipynb` notebook **with outputs committed** (it must run end-to-end and show results). It integrates: LLM, embeddings, vector DB (Chroma/FAISS/Qdrant/Pinecone), agentic RAG, and a chatbot over:
1. A dataset chosen from Kaggle (a theme: movies, books, recipes, etc.).
2. The course-provided long documents ("Obligatorio setiembre 2026 - Papers"), processed via chunking + embeddings + semantic search.

## Hard constraints (graded)

- **Do NOT use `create_agent()`, `create_react_agent`, or any high-level abstraction that hides the agent loop.** Build the graph explicitly with `StateGraph`, nodes, conditional edges, and tool-calling nodes.
- Short-term memory per conversation; multiple independent conversations that can be paused/resumed without losing context (→ persistent checkpointer keyed by `thread_id`).
- Long-term memory shared across conversations for the same user; the agent itself decides what to store and when to use it (→ store keyed by `user_id`).
- Multiple parallel user sessions, uniquely identified.
- Route dynamically between document questions, general questions, and small-talk.
- Hallucination control: answers must be grounded in retrieved context; explicitly say "I don't have enough information" otherwise.
- Tool calling where useful (DB queries, functions, external APIs).
- Evaluation uses unseen questions, so robustness across varied queries matters more than demo polish.

## Required documentation (separate deliverable)

Architecture + justification of choices (LLM, embeddings, vector DB, pipeline, agents); short/long-term memory strategy; prompt examples and prompt-engineering strategies; challenges and solutions; chatbot tests and results; reflection on improvements. Any use of generative AI (including Claude) must be cited: which tool and in what context.

## Optional (extra credit)

Compare embedding models, chunking/retrieval strategies, or LLMs; FastAPI endpoint or Gradio UI.
