# Kaylix Says: A Smart Chatbot Powered by LangGraph and Ollama

**Kaylix Says** is an intelligent chatbot that delivers accurate, natural, and engaging responses to user queries. Built with **LangGraph**, **LangChain**, and **Ollama**, it combines pretrained knowledge with tool-based information retrieval to provide a seamless conversational experience. The chatbot first uses its internal knowledge, then queries an internal knowledge base (`kb` tool), and falls back to web searches via the Tavily API (`tavily` tool) when needed. Its smart routing and transparent explanations make it feel like a true AI companion.

## Features

- **Hierarchical Tool Usage**:
  - Queries an internal knowledge base (`kb`) built from a document (`kb.docx`) using FAISS and Ollama embeddings.
  - Falls back to Tavily web search if the knowledge base returns no relevant results
  - Gracefully handles tool failures with honest responses.
- **Smart Routing**:
  - Uses a custom LangGraph conditional edge to manage the flow between LLM responses and tool calls.
  - Directly checks the `kb` tool’s output for efficient Tavily fallback.
- **Local and Privacy-Focused**:
  - Runs locally with Ollama for LLM and embeddings, ensuring data privacy.
  - Configurable knowledge base via a single document file.

## Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Installed and running locally (default: `http://localhost:11434`) with the `llama3` model pulled.
- **Tavily API Key**: Required for web searches (sign up at Tavily).
- **Dependencies**: Listed in `requirements.txt`.

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/aliasmaya/kaylix_says.git
   cd kaylix_says
   ```

2. **Install Dependencies**: Create a virtual environment and install required packages:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Ollama**:

   - Install Ollama: Follow instructions at Ollama.

   - Pull the `llama3` model:

     ```bash
     ollama pull llama3
     ```

   - Ensure the Ollama server is running:

     ```bash
     ollama serve
     ```

4. **Configure Environment**: Create a `.env` file in the project root:

   ```env
   MODEL=llama3
   BASE_URL=http://localhost:11434
   TAVILY_API_KEY=your-tavily-api-key
   ```

   Replace `your-tavily-api-key` with your Tavily API key.

5. **Prepare Knowledge Base**:

   - Place a `kb.docx` file in the `./data/` directory with your knowledge base content.
   - The chatbot uses this file to build a FAISS vector store for the `kb` tool.
   - Example: `./data/kb.docx` might contain information about specific topics (e.g., "LangGraph is...").

6. **Run the Chatbot**: Start the chatbot:

   ```bash
   python main.py
   ```

   Interact via the command-line interface by typing queries and pressing Enter. Type `quit`, `exit`, or `bye` to stop.

## Dependencies

Listed in `requirements.txt`:

```
langchain
langchain-community
langchain-ollama
langgraph
faiss-cpu
unstructured
python-dotenv
```

Install with:

```bash
pip install -r requirements.txt
```

## Acknowledgments

- Built with LangChain, LangGraph, and Ollama.
- Web search powered by Tavily.

---

Happy chatting with **Kaylix Says**! For questions or feedback, open an issue or contact the maintainers.
