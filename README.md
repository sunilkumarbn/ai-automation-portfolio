# AI & Python Automation Portfolio

A collection of end-to-end Generative AI, Retrieval-Augmented Generation (RAG), and document workflow automation projects built with Python.

---

## Projects Overview

### 1. Agentic RAG & Tool-Calling Assistant (`agentic_rag_assistant/`)
An interactive document intelligence assistant with a Streamlit interface that ingests multi-format files and answers queries grounded in uploaded knowledge bases.
* **Key Features:**
  * Multi-file ingestion supporting **PDF, TXT, and Markdown (`.md`)** documents.
  * Automated text chunking and local vector indexing using **FAISS**.
  * Stateful conversational interface with agentic tool-calling capabilities.
* **Entry Point:** `agentic_rag_assistant/app.py`

---

### 2. Invoice Automation Agent (`invoice_automation_agent/`)
An automated document-processing pipeline that extracts structured data from incoming PDF invoices and logs validated records into an Excel ledger.
* **Key Features:**
  * Automated folder-based state routing (`inbox/` $\rightarrow$ `processed/` or `failed/`).
  * Structured field extraction from PDF invoices into `ledger.xlsx`.
  * Built-in error isolation so malformed files are quarantined without halting the pipeline.
* **Entry Point:** `invoice_automation_agent/agent_pipeline.py`

---

### 3. Research Briefing Pipeline (`research_briefing_pipeline/`)
An automated Python workflow designed to gather, summarize, and structure topic-specific research briefings into clean, digestible outputs.

---

## Tech Stack
* **Language:** Python 3.11+
* **AI & Orchestration:** LangChain, LangGraph, Ollama / LLM APIs, FAISS
* **Frontend & Data:** Streamlit, Pandas, OpenPyXL
* **Automation & File Processing:** PyPDF, Python standard library (`os`, `shutil`, `pathlib`)

---

## Quick Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sunilkumarbn/ai-automation-portfolio.git](https://github.com/sunilkumarbn/ai-automation-portfolio.git)
   cd ai-automation-portfolio