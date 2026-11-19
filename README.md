# Engineering Knowledge Assistant

A local AI-powered assistant that answers questions using engineering documents (PDF and Markdown).

The goal of this project is to understand how RAG (Retrieval-Augmented Generation) works by building it from scratch using Python and open-source tools.

## How It Works

Imagine asking: "How does Redis handle caching?"

Instead of asking an AI model to answer from its general knowledge, we first find relevant information in our own documents and then ask the model to answer using that information.

This approach is called **Retrieval-Augmented Generation (RAG)**.

```text
Documents → Extract Text → Split into Chunks
                                |
                                v
                         Generate Embeddings
                                |
                                v
                           Store in FAISS

User Question → Generate Embedding → Find Relevant Chunks
                                           |
                                           v
                                    Send Context to LLM
                                           |
                                           v
                                     Answer + Sources
```

## Understanding the Components

| Component | What it does | Why we use it |
|---|---|---|
| Document Ingestion | Extracts text from PDF and Markdown files | Makes documents readable by our application |
| Chunking | Splits large documents into smaller pieces | Helps retrieve only relevant information instead of entire documents |
| Embeddings | Converts text into numerical vectors | Allows us to find text with similar meaning, not just matching keywords |
| FAISS | Searches for similar vectors | Quickly finds relevant document chunks |
| Ollama | Runs an LLM locally | Generates answers without paying for hosted AI APIs |
| RAG | Combines retrieval and generation | Grounds answers in our documents and provides sources |

### Important AI Concepts

- **Embedding:** A numerical representation of text that captures aspects of its meaning. Similar meanings generally produce vectors that are close together.
- **Vector Search:** Finds vectors closest to a query vector. We use cosine similarity to compare their direction.
- **Chunk Overlap:** Repeats a small portion of text between adjacent chunks so information near a boundary is less likely to be lost.
- **Hallucination:** When an AI model generates information that is unsupported or incorrect.
- **RAG:** Retrieves relevant information first, then gives that information to an LLM as context for its answer.

## Technology Choices

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| FastAPI | API framework |
| PyMuPDF | Extracts text from PDF files |
| Sentence Transformers | Generates text embeddings |
| FAISS | Stores and searches vectors |
| Ollama + Qwen 3.5 4B | Runs the language model locally |
| Pytest | Tests application behaviour |

## Run Locally

**1. Activate environment**

```bash
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Start Ollama**

```bash
ollama serve
```

**4. Run the RAG pipeline**

In another terminal:

```bash
PYTHONPATH=. python scripts/run_rag.py
```

**5. Run tests**

```bash
pytest -v
```

## Evaluation

Building a RAG system is not enough. We also need to measure whether it retrieves the right information and generates reliable answers.

| Metric | What it tells us |
|---|---|
| Recall@K | Did we retrieve the relevant document within the top K results? |
| MRR | How high did the first relevant result appear in the ranking? |
| Faithfulness | Are the answer's claims supported by the retrieved context? |
| Citation Accuracy | Do the cited sources actually support the claims? |

Faithfulness and citation accuracy currently use an LLM as an evaluator, so their scores are estimates rather than ground truth.

## Experiments

We compared different chunk sizes and embedding models.

**Chunking:** All configurations achieved Recall@3 of 1.0. Chunk sizes of 200 and 500 achieved MRR of 1.0, while 100-character chunks achieved 0.875.

**Embedding models:** MiniLM (384 dimensions) and MPNet (768 dimensions) both achieved Recall@3 and MRR of 1.0 in our current experiment.

We retained MiniLM because it uses fewer dimensions and had lower query evaluation time in our run.

These are preliminary results from a small dataset, not proof of production-level retrieval quality.

## What I've Built

- [x] PDF and Markdown ingestion
- [x] Text chunking
- [x] Embedding generation
- [x] FAISS vector search
- [x] Local LLM integration
- [x] RAG answer generation with source metadata
- [x] Retrieval and answer evaluation
- [x] Chunking and embedding experiments

## Planned Improvements

- [ ] Hybrid search (keyword + vector search)
- [ ] Reranking retrieved results
- [ ] Persistent vector index
- [ ] FastAPI question-answering endpoint
- [ ] Larger evaluation dataset and document collection