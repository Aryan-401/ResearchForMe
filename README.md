## Note: This Script remains untested fully because the Groq Endpoint was continiously giving 503 errors. Sorry for any inconvenience caused. Because of this, I was unable to finish certain features.

# Research Workflow Automation

This project automates the process of conducting research by breaking down queries into smaller questions, searching the web for relevant information, validating results, and generating structured research reports. It leverages tools like vector stores, embeddings, and language models to streamline the workflow.

## Features

- **Query Decomposition**: Breaks down a main query into smaller, researchable questions.
- **Web Search**: Uses DuckDuckGo to search for relevant information.
- **Snippet Validation**: Validates search results based on relevance to the query.
- **Markdown Conversion**: Converts web pages to markdown format for easy storage and retrieval.
- **Research Report Generation**: Creates structured sections for research reports, including sources and content.
- **Workflow Automation**: Implements a state graph to automate the entire research process.

## Project Structure

- `agents/`: Contains logic for query generation, snippet validation, and report writing.
- `graph/`: Implements the workflow using a state graph and nodes for each step.
- `in_memory.py`: Handles in-memory vector store operations for embeddings and similarity searches.
- `try.py`: Example script for testing the workflow.
- `graph/workflow.py`: Main workflow implementation using `StateGraph`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```
     JINA_API_KEY=<your-jina-api-key>
     ```

## Usage

### Running the Workflow

1. Modify the `query` in `graph/workflow.py` to your desired research topic.
2. Run the script:
   ```bash
   python graph/workflow.py
   ```

### Example Output

The workflow will:
1. Generate sub-questions for the query.
2. Search the web for relevant information.
3. Validate the search results.
4. Convert valid links to markdown.
5. Generate structured sections for the research report.

### Customization

- Modify the `Nodes` class in `graph/agent_graph.py` to add or change workflow steps.
- Update the `Tools` class in `agents/tools.py` to integrate additional tools or APIs.

## Dependencies

- Python 3.8+
- [LangChain](https://github.com/hwchase17/langchain)
- [Jina AI](https://jina.ai/)
- DuckDuckGo Search API


## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for providing tools for LLM-based workflows.
- [Jina AI](https://jina.ai/) for embedding and vector store support.
- DuckDuckGo for search capabilities.
```
