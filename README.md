# \# Gutermann Product QA Agent

# 

# A CLI-based Retrieval-Augmented Generation (RAG) question-answering agent for the supplied Gutermann water-leak detection product catalogue.

# 

# \## What it does

# 

# 1\. Reads `product\_overview.md` at startup.

# 2\. Splits the catalogue into variable-sized product-level chunks using its Markdown headings. Relevant parent section context is included with product chunks when applicable.

# 3\. Embeds each chunk locally using `all-MiniLM-L6-v2`.

# 4\. Retrieves relevant product sections using a hybrid score combining semantic similarity (70%) and lexical overlap (30%).

# 5\. Passes the retrieved context to the configured LLM provider (Gemini or Ollama).

# 6\. Instructs the LLM to answer only from the supplied context and explicitly state when the document does not provide enough information.

# 7\. Runs as an interactive CLI application. Type `exit` or `quit` to stop.

# 

# \## Requirements

# 

# \- Python 3.10+

# \- Gemini API key for Gemini-based generation, or Ollama for local generation

# \- Internet access on the first run to download the sentence-transformers model

# \- Internet access for Gemini API calls when using the Gemini provider

# 

# \## Setup on Windows

# 

# Open PowerShell or Command Prompt in the project folder.

# 

# \### 1. Create a virtual environment

# 

# ```powershell

# python -m venv .venv

