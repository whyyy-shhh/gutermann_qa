# \# Gutermann Product QA Agent

# 

# A CLI-based Retrieval-Augmented Generation (RAG) question-answering agent for the supplied Gutermann water-leak detection product catalogue.

# 

# \## What it does

# 

# 1\. Reads `product\\\_overview.md` at startup.

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

# ```

# 

# \### 2. Activate the virtual environment

# 

# PowerShell:

# 

# ```powershell

# .venv\\Scripts\\activate

# ```

# 

# Command Prompt:

# 

# ```cmd

# .venv\\Scripts\\activate.bat

# ```

# 

# \### 3. Install dependencies

# 

# ```powershell

# pip install -r requirements.txt

# ```

# 

# \### 4. Configure the environment

# 

# Copy `.env.example` to `.env`.

# 

# For Gemini:

# 

# ```text

# LLM\_PROVIDER=gemini

# GEMINI\_MODEL=gemini-3.8-flash

# GEMINI\_API\_KEY=your\_api\_key\_here

# ```

# 

# The `.env` file should not be committed to GitHub because it contains the API key.

# 

# \### 5. Run the agent

# 

# ```powershell

# python qa\_agent.py

# ```

# 

# \## Example

# 

# ```text

# Gutermann Product QA Agent

# Type a question, or type 'exit' / 'quit' to stop.

# 

# > What sensors does the AQUASCAN 760T use?

# 

# Agent: The AQUASCAN 760T uses True Sound Sensors (TSS).

# 

# Sources used:

# &#x20; - AQUASCAN 760T (Real-Time Correlators For All Conditions)

# ```

# 

# \## Retrieval Approach

# 

# The system uses hybrid retrieval:

# 

# \- \*\*70% semantic similarity\*\* using normalized embeddings and cosine similarity.

# \- \*\*30% lexical overlap\*\* based on shared query terms.

# 

# The top 5 retrieved chunks are passed to the generation step. A minimum similarity threshold is used to avoid generating answers from clearly irrelevant retrieval results.

# 

# \## Chunking Strategy

# 

# The catalogue is naturally organized using Markdown headings.

# 

# Each `###` product section is treated as a variable-sized chunk. When a product belongs to a parent `##` section containing relevant information, that parent section context is included with the product chunk.

# 

# This preserves product descriptions and features while also retaining information that may be defined at the parent category level.

# 

# \## Why No Vector Database?

# 

# The assignment allows in-memory retrieval, and the supplied catalogue is small enough for this approach.

# 

# Document embeddings are generated at startup and kept in memory for the duration of the process. No cloud vector database is required.

# 

# \## LLM Providers

# 

# The generation layer supports:

# 

# \- \*\*Gemini\*\* for API-based generation.

# \- \*\*Ollama\*\* for local generation.

# 

# The retrieval pipeline is independent of the selected LLM provider.

# 

# \## Test Queries

# 

# The following queries are from the assignment:

# 

# 1\. What is the Minimum Level Profiling feature and which product has it?

# 2\. What is the difference between the AQUASCAN 610 and the AQUASCAN 760T?

# 3\. Which products use True Sound Sensors (TSS) — and what advantage does TSS give?

# 4\. I need to find leaks on plastic pipes over long distances — what do you recommend?

# 5\. We want permanent monitoring in underground chambers with no drilling — what fits?

# 6\. Can I order the ZONESCAN HYDRO today?

# 7\. Does the ZONESCAN AI use hydrophone technology?

# 

# \## Grounding and Hallucination Control

# 

# The generation prompt restricts the LLM to the retrieved catalogue context and instructs it not to invent information.

# 

# If the document does not explicitly confirm something, the agent is instructed to state that rather than making an unsupported claim.

# 

# For example, the ZONESCAN HYDRO ordering question tests whether the system can distinguish documented launch/pre-series information from confirmed current ordering availability.

