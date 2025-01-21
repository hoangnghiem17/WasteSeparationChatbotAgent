import logging
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1) Initiating LLM
logging.info("Initializing LLM with model gpt-4o-mini")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# 2) Defining tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def square(a: int) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

logging.info("Defined tools: add, multiply, square")

# Setup the toolkit
toolkit = [add, multiply, square]

# 3) Define Nodes
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define system prompt for tool calling agent
system_prompt = """
You are a mathematical assistant. You should get the following information from the user:
- mathematical operation to perform
- numbers to perform operations on

If you are not able to discern this information, ask them to clarity! Do not attempt to wildly guess. After you are able to discern all information, use the following tools to answer the query:
- add: Takes in two numbers and returns the sum.
- multiply: Takes in two numbers and returns the product.
- square: Takes in a number and returns the square.

If the user ask for a multiplication, use the add tool the add 10 to the result. In the other cases, just answer the query.

Focus on understanding the user question and solving it with the available tools. If you do not have a tool or knowledge to answer the question, say so. 
Avoid discussing topics that are religious, political, harmful, violent, sexual, filthy, or in any way negative, sad, or provocative.

Here are some examples:
"""

tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tool_runnable = create_tool_calling_agent(llm, toolkit, tool_calling_prompt)
logging.info("Tool calling agent created")

# Define function that will be called when the tool-calling agent is invoked
def run_tool_agent(state):
    agent_outcome = tool_runnable.invoke(state)
    logging.info(f"Agent outcome: {agent_outcome}")
    return {"agent_outcome": agent_outcome}

# Create Tool Executor
from langgraph.prebuilt.tool_executor import ToolExecutor

tool_executor = ToolExecutor(toolkit)
logging.info("Tool executor initialized")

# Define the function to execute tools
def execute_tools(state):
    agent_action = state['agent_outcome']
    if type(agent_action) is not list:
        agent_action = [agent_action]

    steps = []
    for action in agent_action:
        logging.info(f"Executing tool: {action.tool} with input: {action.tool_input}")
        output = tool_executor.invoke(action)
        logging.info(f"Tool result: {output}")
        steps.append((action, str(output)))
    
    return {"intermediate_steps": steps}

# 4) Define Graph State
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, list, ToolAgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[Union[tuple[AgentAction, str], tuple[ToolAgentAction, str]]], operator.add]

# 5) Add Edge Logic
def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        logging.info("Agent finished execution")
        return "END"
    else:
        logging.info("Agent continuing execution")
        return "CONTINUE"

# 6) Build the Graph
from langgraph.graph import END, StateGraph

workflow = StateGraph(AgentState)

workflow.add_node("agent", run_tool_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")

workflow.add_edge('action', 'agent')

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "CONTINUE": "action",
        "END": END
    }
)

app = workflow.compile()
logging.info("Workflow compiled successfully")

# Run the workflow with input
inputs = {"input": "Give me the sum of 10 and 23", "chat_history": []}
config = {"configurable": {"thread_id": "1"}}

for s in app.stream(inputs, config=config):
    output = list(s.values())[0]
    print("------------------------------------------------------")