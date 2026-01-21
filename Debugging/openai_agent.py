from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


#Define State schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
model = ChatGroq(model="openai/gpt-oss-20b")


def make_graph():
    #Add Nodes
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}     
    
    #Initialize graph
    workflow = StateGraph(State)

    #Add Node
    workflow.add_node("agent", call_model)
    
    #Add Edges
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    
    agent = workflow.compile()
    return agent

agent=make_graph()
    