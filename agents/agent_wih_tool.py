from agents.agent import Agents
from agents.tools import Tools, ToolHelpers
from agents.memory import Memory
from dotenv import load_dotenv
from langchain_core.prompts.prompt import PromptTemplate
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain.chat_models import init_chat_model

load_dotenv()


class TheAgent():
    def __init__(self):
        self.agents = Agents()
        self.tool_helpers = ToolHelpers()
        self.tools = Tools(tool_help=self.tool_helpers)
        self.memory = Memory()
        self.model = init_chat_model(model="qwen-qwq-32b", model_provider="groq")
        self.all_tools = self.tools.get_tools()

    def agent_setup(self):
        tools = self.all_tools
        prompt = hub.pull("hwchase17/react")
        print(type(prompt))
        print("prompt", prompt)
        agent = create_react_agent(
            llm=self.model,
            tools=tools,
            prompt=prompt,
            stop_sequence=True,
        )
        return agent

    def run_agent(self, query):
        agent = self.agent_setup()
        agent_executor = AgentExecutor(agent=agent, tools=self.all_tools, verbose=True)
        response = agent_executor.invoke({"input": query})
        return response
