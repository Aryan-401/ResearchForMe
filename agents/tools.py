import json
import random

import requests
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document

from langchain_core.tools import tool
from typing import List, Dict
from agents.agent import Agents
from agents.memory import Memory
import os


class ToolHelpers:
    def __init__(self):
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.vector_store = Memory()
        self.agents = Agents()
        self.judge_db = Memory()

    def get_markdown_from_webpage(self, link):
        """
        Given a link, convert the webpage to markdown.
        """
        headers = {
            "Authorization": "Bearer " + self.jina_api_key,
        }
        response = requests.get(
            url=f"https://r.jina.ai/{link}",
            headers=headers,
        )
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text} for {link}")
            return None
        print(response.text)
        return response.text

    def store_markdown(self, response: str, link: str):
        """
        Given a link, convert the webpage to markdown and store it in a file.
        """
        if response is None:
            print("Error: Response is None")
            return 0
        self.vector_store.save_chunked_memory(
            memory=response,
            metadata={"link": link},
        )
        return 1

    def search_for_sources(self, query: str, k: int = 5) -> List[Document]:
        """
        Given a query, search for sources in the vector store.
        """
        results = self.vector_store.search_recall_memories(query=query, k=k)
        return results

    def create_sections(self, query):
        """
        Given a query, create sections for the research paper.
        """
        return self.agents.create_headings(query)["skeleton"]

    def judge_snippet(self, search_results: List[Dict[str, str]]) -> List[str]:
        links = set()
        for result in search_results:
            query = result["query"]
            results = result["results"]
            for result_ in results:
                title = result_["title"]
                snippet = result_["snippet"]
                self.judge_db.save_recall_memory(
                    memory= title+"_"+snippet,
                    metadata={
                        "link": result_["link"],
                    },
                )
                doc = self.judge_db.get_sim_score(
                    query=query,
                    k=1,
                )[0]
                if doc[1] > 0.7:
                    print("Storing link:", result_["link"])
                    links.add(result_["link"])
                    self.tool_help.store_markdown(
                        self.tool_help.get_markdown_from_webpage(result_["link"]),
                        result_["link"],
                    )
                self.judge_db.delete_id(doc[0].id)

        return list(links)


class Tools:
    def __init__(self, tool_help: ToolHelpers = None):
        self.duckduckgo_search = DuckDuckGoSearchResults(output_format="json", max_results=3)
        self.agent = Agents()
        self.tool_help = tool_help if tool_help else ToolHelpers()

    # @tool
    def search_tool(self, minor_queries: List[str]) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo."""
        search_results = [{"query": query, "results": json.loads(self.duckduckgo_search.run(query))} for query in
                          minor_queries]
        return search_results

    # @tool
    def generate_queries(self, major_query):
        """Given a query, generate a list of questions that will dive deep into the topic and help the user understand it better.
           Your end goal should always be to generate optimized search engine queries that will result in the best research results.
        """
        return self.agent.create_queries(major_query)

    # @tool
    def check_valid_answer(self, search_results: List[Dict[str, str]]) -> List[str]:
        """
        Check if the snippet of the article helps to answer the query.
        """
        links = set()
        for result in search_results:
            query = result["query"]
            results = result["results"]
            for result_ in results:
                title = result_["title"]
                snippet = result_["snippet"]
                try:
                    score = self.agent.judge_snippet(query=query, title=title, snippet=snippet)
                except Exception:
                    print("Error in judging snippet")
                    score = {
                        "score": 0.0,
                        "reasoning": "Error in judging snippet",
                    }
                if score["score"] >= 0.7:
                    # print("Title:", title)
                    # print("Snippet:", snippet)
                    # print("Score:", score["score"])
                    # print("Reasoning:", score["reasoning"])
                    # print("Link:", result_["link"])
                    # print("=" * 100)
                    print("Storing link:", result_["link"])
                    links.add(result_["link"])
                    self.tool_help.store_markdown(
                        self.tool_help.get_markdown_from_webpage(result_["link"]),
                        result_["link"],
                    )

        return list(links)

    # @tool
    def get_sources_for_section(self, query, k):
        """
        Given a query and the max number of sources per question, get the content which you will use to write the section.
        """
        sections = self.tool_help.create_sections(query)
        output = {}
        for section in sections:
            output[f"{section['title']}_{section['level_heading']}"] = []
            for ques in section["questions"]:
                output[f"{section['title']}_{section['level_heading']}"].extend(
                    self.tool_help.search_for_sources(ques, k=k))

        return output, sections

    # @tool
    def write_section(self, sources):
        """
        Given a query and the sources, write the section.
        """
        pass

    def get_tools(self):
        tools = [
            self.search_tool,
            self.generate_queries,
            self.check_valid_answer,
            self.get_sources_for_section,
            self.write_section
        ]
        return tools
