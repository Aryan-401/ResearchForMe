from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langgraph.graph import END, StateGraph

from agents import Agents, ToolHelpers, Tools


class Nodes:

    def __init__(self):
        self.agents = Agents()
        self.tool_helper = ToolHelpers()
        self.tools = Tools(tool_help=self.tool_helper)
        self.model = init_chat_model(model="qwen-qwq-32b", model_provider="groq")

    def get_questions(self, state):
        query = state["query"]
        num_steps = int(state["steps"] + 1)
        minor_questions = self.agents.create_queries(query)

        return {
            "query": query,
            "num_steps": num_steps,
            "research_questions": minor_questions,
        }

    def search_web(self, state):
        queries = state["research_questions"]

        results = self.tools.search_tool(queries)
        return {
            "query": state["query"],
            "num_steps": state["steps"] + 1,
            "research_questions": queries,
            "search_results": results,
        }

    def check_valid_answer(self, state):
        search_results = state["search_results"]
        # valid_links = self.tools.check_valid_answer(search_results)
        valid_links = self.tool_helper.judge_snippet(search_results)
        return {
            "query": state["query"],
            "num_steps": state["steps"] + 1,
            "valid_links": valid_links,
        }

    def get_markdown(self, state):
        valid_links = state["valid_links"]
        markdown = []
        for link in valid_links:
            f = self.tool_helper.get_markdown_from_webpage(link)
            if f is None:
                print(f"Error in converting {link} to markdown")
                continue
            self.tool_helper.store_markdown(
                f, link
            )
            markdown.append(f)
        return {
            "query": state["query"],
            "num_steps": state["steps"] + 1,
        }

    def sources_and_sections(self, state):
        query = state["query"]
        num_steps = state["steps"] + 1
        sources, sections = self.tools.get_sources_for_section(
            query, 2
        )
        return {
            "query": query,
            "num_steps": num_steps,
            "sections": sections,
            "sources": sources,
        }

    def write_section(self, state):
        query = state["query"]
        num_steps = state["steps"] + 1
        sources = state["sources"]
        sections = state["sections"]
        for section in sections:
            for value in sources.values():
                section["content"] = self.agents.write_section(value)
                section['source'] = [
                    x.metadata["link"] for x in value
                ]
        print(state)
        return {
            "query": query,
            "num_steps": num_steps,
            "sections": sections,
            "sources": sources,
        }

    def return_markdown(self, state):
        sections = state["sections"]

        def render_sections_as_markdown(sections) -> str:
            markdown_output = []
            for section in sections:
                heading_prefix = '#' * section.level_heading
                markdown_output.append(f"{heading_prefix} {section.title}\n")
                markdown_output.append(f"{section.content.strip()}\n")
                if section.sources:
                    markdown_output.append("### Sources\n")
                    for source in section.sources:
                        markdown_output.append(f"- [{source}]({source})\n")
            return '\n'.join(markdown_output)

        result = render_sections_as_markdown(sections)
        with open(f"{state['query']}.md", "w") as f:
            f.write(result)
        return {
            "query": state["query"],
            "num_steps": state["steps"] + 1,
            "drafts": [result],
        }
