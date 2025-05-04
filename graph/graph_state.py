import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing_extensions import TypedDict
from typing import List, Dict
from agents.structured_output import ResearchQuestions, RelevantSnippet, Skeleton


class GraphState(TypedDict):
    query: str
    search_results: List[Dict[str, str]]
    research_questions: ResearchQuestions
    sources: Dict[str, List[str]]
    sections: Skeleton
    steps: int
    valid_links: List[str]
    drafts: List[str]
