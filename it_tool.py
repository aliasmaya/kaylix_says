from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from it_kb import knowledge_base

tool = TavilySearchResults(max_results=2, name='tavily')
tools = [tool, knowledge_base]
tool_node = ToolNode(tools=tools)