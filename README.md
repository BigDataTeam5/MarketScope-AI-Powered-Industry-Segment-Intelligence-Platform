# MarketScope AI: Healthcare Product Analytics 

## 👥 Team Members
- **Yash Khavnekar** – Data Collection, Web Scraping, Sentiment Analysis (MCP)
- **Shushil Girish** – Agent Integration, Backend + ETL (LangGraph, FastAPI, Airflow)
- **Riya Mate** – Frontend, Marketing Automation,Documentation, Codelabs

## Features

- **Market Segmentation Analysis** - Analyze and understand different healthcare market segments
- **Strategic Query Optimization** - Get optimized answers to your strategic questions
- **Product Comparison** - Compare products across different segments
- **Sales & Marketing Analysis** - Upload your sales data for AI-powered insights

**MarketScope** is a scalable AI-powered market intelligence platform designed to democratize access to industry insights. It integrates **structured data** from Snowflake Marketplace with **real-time unstructured data** like product re pricing from vendors. Informed by foundational marketing literature (e.g., *Philip Kotler’s Marketing Management*), the platform delivers pricing analysis, sentiment breakdowns, and market sizing (TAM/SAM/SOM) to help startups, SMBs, and analysts explore opportunities without costly reports or consultants.

### Prerequisites

- Python 3.8 or higher
- Required packages (installed automatically when following the setup instructions)

### Setup and Running
![diagram-export-4-18-2025-2_51_50-PM](https://github.com/user-attachments/assets/6f26c242-6cfc-4f3a-9d49-d6fafab3aa03)

#### Simple Method
Run the simplified starter script:

---

## 🔧 Architecture Overview

![diagram-export-4-18-2025-2_51_50-PM](https://github.com/user-attachments/assets/b71bc874-d979-4f89-8c0a-8597c17368d5)


- **Frontend**: Streamlit (exploratory dashboard)
- **Backend**: FastAPI
- **Agents**: LangGraph + MCP (Model Context Protocol)
- **ETL Pipelines**: Apache Airflow
- **Data Sources**:
  - Structured: [Snowflake Marketplace – Industry & Segment Data](https://app.snowflake.com/)
  - Unstructured: Web scraping (reviews) + Web search (pricing)

---
## File Structure
```
MarketScope-AI-Powered-Industry-Segment-Intelligence-Platform/
│
├── Airflow/                            # Airflow pipeline components
│   ├── dags/                           # Airflow DAGs
│   │   └── book_to_vector_pipeline.py  # PDF processing pipeline
│   ├── config/                         # Configuration files
│   │   └── book.json                   # Book processing config
│   └── utils/                          # Utility modules
│       ├── mistralparsing_userpdf.py   # PDF parsing
│       └── chunking.py                 # Text chunking utilities
│
├── mcp_server/                         # Master Control Program server
│   ├── __init__.py
│   ├── server.py                       # Main MCP server implementation
│   ├── config.py                       # Server configuration
│   ├── models.py                       # Data models for MCP
│   ├── utils/                          # MCP utilities
│   │   ├── __init__.py
│   │   ├── auth.py                     # Authentication utilities
│   │   └── logging.py                  # Logging configuration
│   └── services/                       # Core MCP services
│       ├── __init__.py
│       ├── session_manager.py          # Session management
│       ├── agent_registry.py           # Agent registration/discovery 
│       └── task_queue.py               # Task scheduling
│
├── agents/                             # Specialized agents
│   ├── __init__.py
│   ├── base_agent.py                   # Base agent class
│   ├── analysis_agent/                 # Analysis agent
│   │   ├── __init__.py
│   │   ├── server.py                   # Analysis agent server
│   │   └── analyzers/                  # Analysis modules
│   │       └── market_analyzer.py      # Market analysis
│   │
│   ├── research_agent/                 # Research agent
│   │   ├── __init__.py
│   │   ├── server.py                   # Research agent server
│   │   └── knowledge_base.py           # Knowledge retrieval
│   │
│   └── marketing_agent/                # Marketing agent
│       ├── __init__.py
│       ├── server.py                   # Marketing agent server
│       └── generators/                 # Content generation
│           └── content_generator.py    # Marketing content
│
├── client/                             # Client applications
│   ├── cli/                            # Command line interface
│   │   └── marketscope_cli.py          # CLI tool
│   └── web/                            # Web interface
│       ├── app.py                      # Web app
│       ├── static/                     # Static assets
│       └── templates/                  # HTML templates
│
├── setup_pinecone.py                   # Pinecone setup script
├── requirements.txt                    # Dependencies
├── .env.example                        # Example environment variables
├── README.md                           # Project documentation
└── docker-compose.yml                  # Container orchestration
```

This will:
1. Check for required packages and install them if needed
2. Start the API server
3. Start the Streamlit frontend
4. Provide you with the URL to access the application

#### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
python api/main.py
```

3. Start the Streamlit frontend:
```bash
streamlit run frontend/app.py
```

## Project Structure

- `api/` - FastAPI backend server
- `frontend/` - Streamlit user interface
- `mcp_servers/` - Model Context Protocol (MCP) servers for different functionalities
- `agents/` - AI agents for analysis
- `config/` - Application configuration

- **Frontend + Backend**: GCP Cloud Run (containerized with Docker)
- **Pipelines**: Cloud Composer (Airflow DAG orchestration)
- **CI/CD**: GitHub Actions for pipeline updates and app deployment
- **Secrets & Cache**:  GCP Secret Manager

The platform supports analysis across these healthcare segments:

- Diagnostic Segment
- Supplement Segment
- OTC Pharmaceutical Segment
- Fitness Wearable Segment
- Skin Care Segment

## Data Analysis

To analyze your sales data:

1. Select your segment from the sidebar
2. Go to Sales & Marketing Analysis page
3. Upload your CSV file or use the sample data
4. Click "Analyze Data"

- Democratize access to industry research insights
- Automate market sizing (TAM/SAM/SOM) and tier classification
- Provide book-informed strategic Q&A based on marketing theory

MarketScope includes a Retrieval Augmented Generation (RAG) system that provides access to marketing knowledge from Philip Kotler's Marketing Management book:

1. Go to the Query Optimization page
2. Enter your marketing question
3. The system will retrieve relevant sections from the book
4. Get tailored marketing strategies for your specific segment

## Deployed Links:
-## MCP SERVER:
- unified server - http://34.42.74.104:8000/docs
- marketing analysis server - http://34.42.74.104:8001/docs
- snowflake mcp server - http://34.42.74.104:8004/docs
- sales data analysis server - http://34.42.74.104:8002/docs
- segment mcp server - http://34.42.74.104:8003/docs

## Troubleshooting

If you encounter issues:

- Check that all required dependencies are installed
- Verify that no other applications are using the required ports (8000-8004, 8501)
- Ensure your environment variables are properly set up
- Check the application logs for specific error messages

## CodeLabs
https://codelabs-preview.appspot.com/?file_id=1_936snjPYvoj-RmfO5Vcm2G8xzjVTv0XGRy5wHlFiCo/edit?pli=1&tab=t.0#0

## License

[MIT License](LICENSE)
