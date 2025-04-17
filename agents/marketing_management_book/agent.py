"""
Marketing Management Agent using LangChain for healthcare market segmentation
"""
import logging
from typing import Dict, Any, Optional, List
# Fix #1: Update import to use langchain_openai instead of deprecated import
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langsmith import traceable
from config import Config

# Initialize logging
logger = logging.getLogger('marketing_agent')

class MarketingManagementAgent:
    def __init__(self):
        self.tools = None
        self.agent = None
        self.current_segment = None
        self.llm = None

    def convert_to_langchain_tools(self, tool_functions):
        """Convert tool functions to LangChain Tool objects"""
        tools = []
        for name, func in tool_functions.items():
            tool = Tool(
                name=name,
                func=func,
                description=func.__doc__ or f"Tool for {name}",
                return_direct=False
            )
            tools.append(tool)
        return tools

    @traceable(name="setup_agent", run_type="chain")
    async def setup(self):
        """Setup agent with tools"""
        try:
            # Load tools
            from .marketing_tools import tool_functions
            
            # Convert to LangChain tools
            self.tools = self.convert_to_langchain_tools(tool_functions)
            print(f"Converted {len(self.tools)} tools to LangChain format")
            
            # Fix #1: Updated ChatOpenAI import above
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=Config.OPENAI_API_KEY
            )
            
            # Create a compliant ReAct prompt template
            # ReAct requires {tools} and {agent_scratchpad} variables
            prompt = ChatPromptTemplate.from_template("""You are a healthcare market analyst helping with the {segment} segment.

Available tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}""")
            
            # Create agent
            agent = create_react_agent(self.llm, self.tools, prompt)
            
            # Create executor with appropriate parameters
            self.agent = AgentExecutor(
                agent=agent, 
                tools=self.tools, 
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3
            )
            
            print("Successfully created LangChain agent with tools")
            return True
            
        except Exception as e:
            print(f"Error setting up agent: {str(e)}")
            return False

    @traceable(name="process_query", run_type="chain")
    async def process_query(self, query, segment=None):
        """Process a query using the agent"""
        try:
            if segment:
                self.current_segment = segment
            
            # Initialize if needed
            if not self.agent and not await self.setup():
                return {"error": "Agent initialization failed", "status": "error"}

            # Get tool names for the prompt
            tool_names = ", ".join([tool.name for tool in self.tools])
            
            # Get tool descriptions for the prompt
            tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            
            # Invoke with all required parameters for ReAct
            response = await self.agent.ainvoke({
                "input": query,
                "segment": self.current_segment or "Healthcare",
                "tool_names": tool_names,
                "tools": tool_descriptions,
                "agent_scratchpad": []  # Already fixed to empty list
            })

            return {
                "output": response.get("output", str(response)),
                "status": "success",
                "segment": self.current_segment
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error",
                "segment": self.current_segment
            }

# Create singleton instance
marketing_agent = MarketingManagementAgent()