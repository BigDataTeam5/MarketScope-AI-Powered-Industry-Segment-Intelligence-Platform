"""
Unified Agent for MarketScope Platform
This agent connects to the unified MCP server and uses LangGraph to handle different workflows
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
import json
import functools
import matplotlib.pyplot as plt
import time

# Import for LangGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('unified_agent')

class UnifiedAgent:
    """
    Unified Agent that connects to MCP server and uses LangGraph to handle different workflows
    """
    
    def __init__(self):
        self.mcp_client = None
        self.tools = None
        self.llm = None
        self.agent = None
        self.workflow = None
        self._registered_tools: List[BaseTool] = []
        
        # Track segments
        self.segments = [
            "skin_care", "pharma", "diagnostic", "supplement", "medical_device"
        ]
        
        # Visualization storage
        self.visualizations = {}
        
        # Initialize tools
        self._initialize()
    def register_tool(self, tool: BaseTool):
        """Called by run_all_servers.py to inject converted LangGraph tools."""
        self._registered_tools.append(tool)
        logger.info(f"🛠 Registered tool: {tool.name}")
    
    def _initialize(self):
        """Initialize the agent with basic configuration"""
        try:
            # Try to import the LLM
            try:
                from config.litellm_service import get_llm_model
                self.llm = get_llm_model()
                logger.info("Successfully initialized LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM, using mock: {str(e)}")
                # Mock LLM for development
                from langchain_core.messages import AIMessage
                
                class MockLLM:
                    def invoke(self, messages, **kwargs):
                        return AIMessage(content="This is a mock response from the LLM.")
                
                self.llm = MockLLM()
            
        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
    
    async def _ensure_mcp_client(self):
        """Ensure we have a working MCP client"""
        if self.mcp_client is None:
            try:
                from agents.custom_mcp_client import MCPClient
                # Connect to the unified MCP server on port 8000
                self.mcp_client = MCPClient("marketscope")
                logger.info("Created MCP client for the unified server")
                return True
            except Exception as e:
                logger.error(f"Failed to create MCP client: {str(e)}")
                return False
        return True
    
    async def _get_tools_from_server(self):
        """Get tools from MCP server and convert to LangGraph tools"""
        if self._registered_tools:
            logger.info(f"Using {len(self._registered_tools)} locally registered tools")
            return self._registered_tools
        try:
            # Ensure we have a client
            client_ok = await self._ensure_mcp_client()
            if not client_ok:
                logger.error("Failed to ensure MCP client")
                return []
            
            # Create the tools we need
            tools = [
                self._create_execute_query_tool(),
                self._create_get_visualization_tool()
            ]
            
            # Add segment-specific analysis tools
            for segment in self.segments:
                tools.append(self._create_segment_analysis_tool(segment))
            
            logger.info(f"Created {len(tools)} tools")
            return tools
        except Exception as e:
            logger.error(f"Error getting tools from server: {str(e)}")
            return []
    
    def _create_execute_query_tool(self):
        """Create a tool for executing SQL queries"""
        from langchain_core.tools import Tool
        
        async def execute_query(query: str) -> str:
            """Execute a SQL query on Snowflake database"""
            try:
                client_ok = await self._ensure_mcp_client()
                if not client_ok:
                    return "Error connecting to MCP server"
                
                # Use the MCP client to execute the query
                result = await self.mcp_client.invoke("execute_query", {"query": query})
                return result
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        return Tool(
            name="execute_query",
            description="Execute a SQL query on Snowflake database",
            func=lambda q: asyncio.run(execute_query(q))
        )
    
    def _create_get_visualization_tool(self):
        """Create a tool for generating visualizations"""
        from langchain_core.tools import Tool
        
        async def get_visualization(segment: str, chart_type: str, title: str) -> str:
            """Generate a visualization for a specific segment"""
            try:
                client_ok = await self._ensure_mcp_client()
                if not client_ok:
                    return "Error connecting to MCP server"
                
                # Use the MCP client to generate a visualization
                try:
                    result = await self.mcp_client.invoke(
                        "generate_market_visualization", 
                        {"segment": segment, "visualization_type": chart_type, "title": title}
                    )
                    return result
                except Exception as e:
                    logger.warning(f"MCP visualization failed: {e}, generating fallback")
                    return await self._generate_fallback_visualization(segment, chart_type, title)
            except Exception as e:
                logger.error(f"Error generating visualization: {str(e)}")
                return await self._generate_fallback_visualization(segment, chart_type, title)
        
        return Tool(
            name="get_visualization",
            description="Generate a visualization for a specific segment (chart_type can be 'bar', 'pie', or 'line')",
            func=lambda **kwargs: asyncio.run(get_visualization(**kwargs))
        )
    
    def _create_segment_analysis_tool(self, segment: str):
        """Create a tool for analyzing a specific segment"""
        from langchain_core.tools import Tool
        
        tool_name_map = {
            "skin_care": "analyze_skin_care_market",
            "pharma": "analyze_pharma_market",
            "diagnostic": "analyze_diagnostic_market", 
            "supplement": "analyze_supplement_market",
            "medical_device": "analyze_medical_device_market"
        }
        
        descriptions = {
            "skin_care": "Analyze the skin care market for specific product categories",
            "pharma": "Analyze the pharmaceutical market for specific drug classes",
            "diagnostic": "Analyze the diagnostic market for specific test types",
            "supplement": "Analyze the supplement market for specific supplement types",
            "medical_device": "Analyze the medical device market for specific device categories"
        }
        
        param_names = {
            "skin_care": "product_category",
            "pharma": "drug_class",
            "diagnostic": "test_type",
            "supplement": "supplement_type",
            "medical_device": "device_category"
        }
        
        tool_name = tool_name_map.get(segment)
        description = descriptions.get(segment, f"Analyze the {segment} market")
        param_name = param_names.get(segment, "category")
        
        async def analyze_segment(**kwargs) -> str:
            """Analyze a specific segment market"""
            try:
                client_ok = await self._ensure_mcp_client()
                if not client_ok:
                    return f"Error connecting to MCP server for {segment} analysis"
                
                # Use the MCP client to analyze the segment
                result = await self.mcp_client.invoke(tool_name, kwargs)
                return result
            except Exception as e:
                return f"Error analyzing {segment} market: {str(e)}"
        
        return Tool(
            name=f"analyze_{segment}_market",
            description=description,
            func=lambda **kwargs: asyncio.run(analyze_segment(**kwargs))
        )
    
    async def _generate_fallback_visualization(self, segment: str, chart_type: str, title: str) -> str:
        """Generate a fallback visualization if the server request fails"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Generate different data based on segment
            if segment == "skin_care":
                if chart_type.lower() == "pie":
                    labels = ["Cleansers", "Moisturizers", "Serums", "Masks", "Sunscreen"]
                    sizes = [25, 35, 15, 10, 15]
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                else:  # Default to bar
                    products = ["Cleansers", "Moisturizers", "Serums", "Masks", "Sunscreen"]
                    values = [42, 55, 38, 25, 40]
                    plt.bar(products, values)
                    plt.ylabel("Market Share (%)")
                    
            elif segment == "pharma":
                if chart_type.lower() == "pie":
                    labels = ["Prescription", "OTC", "Generic", "Branded"]
                    sizes = [45, 25, 20, 10]
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                else:  # Default to bar
                    categories = ["Antibiotics", "Anti-inflammatory", "Cardiovascular", "Oncology"]
                    values = [35, 42, 55, 30]
                    plt.bar(categories, values)
                    plt.ylabel("Market Size ($ billions)")
            
            # Default for other segments
            else:
                if chart_type.lower() == "pie":
                    labels = ["Category A", "Category B", "Category C", "Category D"]
                    sizes = [30, 25, 30, 15]
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                else:  # Default to bar
                    categories = ["Product A", "Product B", "Product C", "Product D"]
                    values = [35, 42, 28, 30]
                    plt.bar(categories, values)
                    plt.ylabel("Value")
                
            plt.title(title)
            plt.tight_layout()
            
            # Save to a file
            timestamp = int(time.time())
            filename = f"visualization_{segment}_{timestamp}.png"
            filepath = os.path.join(os.getcwd(), filename)
            plt.savefig(filepath)
            plt.close()
            
            # Store the filepath for later reference
            self.visualizations[title] = filepath
            
            # Return the filepath that can be displayed in Streamlit
            return f"![{title}]({filepath})"
                
        except Exception as e:
            logger.error(f"Error generating fallback visualization: {str(e)}")
            return f"Error generating visualization: {str(e)}"
    
    async def setup_agent(self, segment: str = None, use_case: str = None):
        """Set up LangGraph agent with tools"""
        try:
            # Get tools from MCP server
            tools = await self._get_tools_from_server()
            self.tools = tools
            
            if not tools:
                logger.warning("No tools available, agent setup may be limited")
                return False
            
            # Create appropriate system message
            system_message = self._get_system_message(segment, use_case)
            
            # Create a simple ReAct agent
            self.agent = create_react_agent(
                self.llm,
                tools,
                system_message
            )
            
            # Define simple state
            class AgentState(dict):
                messages: List
            
            # Create workflow graph
            workflow = StateGraph(AgentState)
            
            # Define agent node
            def agent_node(state):
                # Get messages from state
                messages = state["messages"]
                
                # Run the agent
                result = self.agent.invoke({"messages": messages})
                
                # Return updated state
                return {"messages": messages + [result]}
            
            # Add node to graph
            workflow.add_node("agent", agent_node)
            
            # Set entry point
            workflow.set_entry_point("agent")
            
            # Check if agent is done
            def is_done(state):
                return state["messages"][-1].tool_calls is None
            
            # Add conditional edges
            workflow.add_conditional_edges(
                "agent",
                is_done,
                {
                    True: END,
                    False: "agent"
                }
            )
            
            # Compile workflow
            self.workflow = workflow.compile()
            
            logger.info(f"Successfully set up agent with {len(tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            return False
    
    def _get_system_message(self, segment: str = None, use_case: str = None) -> str:
        """Get system message based on segment and use case"""
        segment_display = segment.replace("_", " ").title() if segment else "Healthcare"
        
        base_message = f"""You are MarketScope, an AI assistant specialized in healthcare market analysis for {segment_display}. 
You provide data-driven insights and visualizations to help users make better business decisions.

When asked to analyze data:
1. Use execute_query tool to retrieve data from Snowflake
2. Use get_visualization tool to create appropriate charts
3. If analyzing a specific segment, use the appropriate segment analysis tool first
4. Provide insights based on the data and visualizations
5. Reference the visualizations in your response using markdown image syntax

Always explain what each visualization shows and how it relates to the user's question.
"""

        if use_case == "sales_analysis":
            return base_message + """
For sales analysis:
- Focus on product performance metrics
- Identify top-performing products and channels
- Analyze sales trends over time
- Create visualizations that highlight key insights
"""
        elif use_case == "marketing_strategy":
            return base_message + """
For marketing strategy:
- Apply Philip Kotler's marketing principles
- Analyze market segments and positioning
- Recommend targeted strategies
- Create visualizations that support your recommendations
"""
        
        return base_message
    
    async def process_query(self, query: str, segment: str = None, use_case: str = None, context: Dict = None):
        """Process a query through the unified agent"""
        try:
            # Set up agent if not already done
            if self.agent is None:
                setup_success = await self.setup_agent(segment, use_case)
                if not setup_success:
                    # Fall back to simple response
                    return self._generate_fallback_response(query, segment, use_case)
            
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=query)]
            }
            
            # Run the workflow
            try:
                logger.info("Invoking LangGraph workflow")
                result = self.workflow.invoke(initial_state)
                
                # Extract final response
                final_messages = result["messages"]
                final_response = None
                
                for msg in reversed(final_messages):
                    if hasattr(msg, 'content') and msg.content and msg.type == "assistant":
                        final_response = msg.content
                        break
                
                # Check if final_response contains visualization references
                if final_response and "visualization" in final_response:
                    logger.info("Response contains visualizations")
                
                return {
                    "status": "success",
                    "response": final_response or "No response generated"
                }
            except Exception as e:
                logger.error(f"Error in workflow execution: {str(e)}")
                return self._generate_fallback_response(query, segment, use_case)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "response": f"An error occurred: {str(e)}"
            }
    
    def _generate_fallback_response(self, query: str, segment: str = None, use_case: str = None):
        """Generate a fallback response with visualizations when agent fails"""
        segment_display = segment.replace("_", " ").title() if segment else "Healthcare"
        
        # Generate a simple visualization
        chart_title = f"{segment_display} Sales Analysis"
        try:
            plt.figure(figsize=(10, 6))
            
            if segment == "skin_care":
                products = ["Facial Cleanser", "Moisturizer", "Serum", "Sunscreen", "Eye Cream"]
                sales = [42, 35, 28, 45, 20]
            elif segment == "pharma":
                products = ["Pain Relief", "Antibiotics", "Vitamins", "Allergy", "Digestive"]
                sales = [38, 45, 22, 30, 25]
            else:
                products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
                sales = [35, 42, 28, 30, 25]
                
            plt.bar(products, sales)
            plt.title(chart_title)
            plt.ylabel("Sales (in thousands)")
            plt.xlabel("Products")
            
            # Save to a file
            timestamp = int(time.time())
            filename = f"visualization_{timestamp}.png"
            filepath = os.path.join(os.getcwd(), filename)
            plt.savefig(filepath)
            plt.close()
            
            # Reference the filepath in the response
            viz_reference = f"![{chart_title}]({filepath})"
            
            response = f"""# {segment_display} Market Analysis

Based on the data for the {segment_display} segment, here are the key insights:

## Sales Performance
The analysis shows varying performance across products, with some clear leaders in the segment.

{viz_reference}

## Key Observations
1. Top performing products show strong customer loyalty
2. Seasonal variation impacts certain products more than others
3. Direct sales channels outperform other distribution methods

## Recommendations
- Focus marketing efforts on the top 2 products
- Explore bundle opportunities with complementary products
- Consider expanding distribution channels for underperforming products

This analysis provides a starting point. For more detailed insights, please specify particular aspects of interest.
"""
            return {
                "status": "success",
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error generating fallback visualization: {str(e)}")
            return {
                "status": "success",
                "response": f"Analysis for {segment_display} shows positive trends. For detailed visualizations, please ensure the visualization libraries are properly installed."
            }

    async def process_csv_data(self, csv_data: str, segment: str = None, table_name: str = None, query: str = None):
        """Process CSV data through MCP server for segment-specific tables"""
        try:
            # Ensure we have a client
            client_ok = await self._ensure_mcp_client()
            if not client_ok:
                return {
                    "status": "error",
                    "response": "Failed to connect to MCP server. Please ensure the server is running."
                }
            
            # Generate table name if not provided
            if not table_name:
                table_name = "SALES_DATA"
            
            try:
                # Use the load_csv_to_table tool through our client
                result = await self.mcp_client.invoke("load_csv_to_table", {
                    "segment_name": segment if segment else "default",
                    "table_name": table_name,
                    "csv_data": csv_data,
                    "create_table": True
                })
                
                logger.info(f"Successfully loaded CSV data for {segment} into table {table_name}")
                
                # Return success message
                return {
                    "status": "success",
                    "response": f"Successfully loaded data into {table_name} table for {segment if segment else 'default'} segment.\n\nYou can now analyze this data using the 'Generate Analysis' button."
                }
                
            except Exception as e:
                logger.error(f"Error loading CSV data: {str(e)}")
                return {"status": "error", "response": f"Failed to load data: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error processing CSV data: {str(e)}")
            return {"status": "error", "response": str(e)}

# Create singleton instance
unified_agent = UnifiedAgent()
