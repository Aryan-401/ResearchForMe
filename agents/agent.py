from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from agents.structured_output import ResearchQuestions, RelevantSnippet, Skeleton
from typing import List, Dict, Any


class Agents:
    def __init__(self):
        self.query_maker = init_chat_model(model="meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq")
        self.judge = init_chat_model(model="llama3-70b-8192", model_provider="groq")
        self.writer = init_chat_model(model="compound-beta", model_provider="groq")

    def create_queries(self, query) -> List[str]:
        # messages = [
        #     (
        # "system", """
        # You are an intelligent assistant that generates additional questions that will better the research process.
        # Given a query, generate a list of questions that will dive deep into the topic and help the user understand it better.
        # Your end goal should always be to generate optimized search engine queries that will result in the best research results.
        # """
        #     )
        # ]
        # template = ChatPromptTemplate(messages)
        structured_llm = self.query_maker.with_structured_output(ResearchQuestions)
        answer = structured_llm.invoke(query)
        return answer.model_dump()["questions"]

    # def judge_snippet(self, query, title, snippet):
    #     # messages = [
    #     #     (
    #     #         "system", """
    #     #         You are an intelligent assistant that judges whether a snippet of an article helps to answer the query.
    #     #         Given a query, title, and snippet, determine if the snippet is relevant to the query.
    #     #         If the snippet is relevant, return a score of 1.0. Only share the score,
    #     #         """
    #     #     )
    #     # ]
    #     structured = self.judge.with_structured_output(RelevantSnippet)
    #     answer = structured.invoke(f"""Title: {title}
    #     Snippet: {snippet}
    #     Query: {query}
    #     """)
    #     return answer.model_dump()

    def create_headings(self, query):
        structured_llm = self.query_maker.with_structured_output(Skeleton)
        answer = structured_llm.invoke(query)
        return answer.model_dump()

    def write_section(self, source: List[Document]):
        messages = [
            (
                "system", """
                You are an intelligent assistant that writes a section of a research report based on the provided query and sources.
                Use the sources to write the section and ensure that the content is relevant. Write the section in a clear and concise manner, do not make up facts or hallucinate.
                But you can use your own knowledge to fill in the gaps. Your answer must be in markdown format.
                """
            ),
        ]

        for s in source:
            messages.append(("human",
                             f"""
            Source: {s.page_content}
            """
                             ))
        template = ChatPromptTemplate(messages)
        answer = template | self.writer
        return answer.invoke(input={"query": "Write the section based on the sources."}).content
