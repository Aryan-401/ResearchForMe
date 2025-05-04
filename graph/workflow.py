from langgraph.graph import END, StateGraph
from graph_state import GraphState
from agent_graph import Nodes
from dotenv import load_dotenv


class Workflow():
    def __init__(self):
        load_dotenv()
        self.workflow = StateGraph(GraphState)
        self.nodes = Nodes()
        self.workflow.add_node("get_questions", self.nodes.get_questions)
        self.workflow.add_node("search_web", self.nodes.search_web)
        self.workflow.add_node("chec_valid_answer", self.nodes.check_valid_answer)
        self.workflow.add_node("get_markdown", self.nodes.get_markdown)
        self.workflow.add_node("source", self.nodes.sources_and_sections)
        self.workflow.add_node("write_section", self.nodes.write_section)
        self.workflow.add_node("convert_to_markdown", self.nodes.return_markdown)

        self.workflow.set_entry_point("get_questions")
        self.workflow.add_edge("get_questions", "search_web")
        self.workflow.add_edge("search_web", "chec_valid_answer")
        self.workflow.add_edge("chec_valid_answer", "get_markdown")
        self.workflow.add_edge("get_markdown", "source")
        self.workflow.add_edge("source", "write_section")
        self.workflow.add_edge("write_section", "convert_to_markdown")
        self.workflow.add_edge("convert_to_markdown", END)

        self.app = self.workflow.compile()

    def run(self, state: GraphState):
        """
        Run the workflow with the given state.
        """
        output = self.app.invoke(state)
        return output


if __name__ == "__main__":
    from pprint import pprint
    from graph.graph_state import GraphState
    from agents.structured_output import ResearchQuestions, Skeleton

    state = GraphState(
        query="Discovery of Electricity",
        search_results=[],
        research_questions=ResearchQuestions(
            questions=["What is the capital of France?"],
            answers=["Paris"]
        ),
        sources={},
        sections=Skeleton(),
        steps=0,
        drafts=[],
        valid_links=[],
    )

    workflow = Workflow()
    output = workflow.run(state)
    pprint(output)
