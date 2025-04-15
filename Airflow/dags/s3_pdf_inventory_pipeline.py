"""
S3 PDF Inventory Pipeline

Lists all PDFs in the S3 bucket organized by:
- Industry (e.g., healthcare industry)
- Segment (e.g., Diagnostic segment)
- Company (e.g., Abbott Laboratories)

Also identifies common reports within each segment.

This pipeline:
1. Lists all PDFs in the specified S3 bucket
2. Organizes them by industry/segment/company
3. Identifies common reports within segments
4. Stores the results as JSON in S3
"""
import os
import json
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.exceptions import AirflowFailException
from airflow.models import Variable

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)

# Configuration
AWS_CONN_ID = "aws_default"  # Change to match your Airflow connection ID
S3_BUCKET = "finalproject-product"  # Your bucket name
OUTPUT_PREFIX = "inventory/reports"  # Where to store inventory reports

# Default arguments for DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 9),
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}

# Constants
INDUSTRY_SEGMENTS = [
    "healthcare industry",
    # Add other industries as needed
]

def extract_path_components(s3_key: str) -> Optional[Dict[str, str]]:
    """
    Extract industry, segment, (optionally company), and filename from S3 key.
    We handle both:
      3-part: industry/segment/filename.pdf
      4-part: industry/segment/company/filename.pdf
    """
    parts = s3_key.split('/')
    # Always ignore anything that isn't a .pdf at the final part
    if not parts[-1].lower().endswith('.pdf'):
        return None
    
    if len(parts) == 3:
        # e.g. healthcare industry/Diagnostic segment/segmentstudy.pdf
        return {
            "industry": parts[0],
            "segment": parts[1],
            "company": None,   # No company folder
            "filename": parts[2]
        }
    elif len(parts) == 4:
        # e.g. healthcare industry/Diagnostic segment/Abbott Laboratories/0001628280.pdf
        return {
            "industry": parts[0],
            "segment": parts[1],
            "company": parts[2],
            "filename": parts[3]
        }
    else:
        # If there's more or fewer than 3 or 4 components, we ignore
        return None

def list_pdfs_in_s3(**context) -> Dict:
    logging.info(f"Starting to list PDFs in S3 bucket: {S3_BUCKET}")
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    inventory = {
        "timestamp": datetime.now().isoformat(),
        "bucket": S3_BUCKET,
        "industries": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    total_pdfs = 0
    
    for industry in INDUSTRY_SEGMENTS:
        logging.info(f"Listing PDFs for industry: {industry}")
        
        # 1) Remove delimiter
        industry_keys = s3_hook.list_keys(
            bucket_name=S3_BUCKET,
            prefix=f"{industry}/"   # no delimiter
        )
        
        for key in industry_keys or []:
            # 2) Parse path
            components = extract_path_components(key)
            if not components:
                continue
            
            # 3) Handle None vs actual company
            if components["company"] is None:
                company_key = "__common__"
            else:
                company_key = components["company"]
            
            # 4) Insert into inventory
            inventory["industries"][components["industry"]] \
                     [components["segment"]] \
                     [company_key].append(components["filename"])
            total_pdfs += 1
            
            if total_pdfs % 50 == 0:
                logging.info(f"Processed {total_pdfs} PDFs so far...")
    
    # Convert your defaultdict into normal dict
    inventory_dict = {
        "timestamp": inventory["timestamp"],
        "bucket": inventory["bucket"],
        "industries": {}
    }
    for ind, seg_dict in inventory["industries"].items():
        inventory_dict["industries"][ind] = {}
        for seg, comp_dict in seg_dict.items():
            inventory_dict["industries"][ind][seg] = dict(comp_dict)
    
    # XCom push
    context['ti'].xcom_push(key='pdf_inventory', value=inventory_dict)
    context['ti'].xcom_push(key='total_pdfs', value=total_pdfs)
    logging.info(f"Completed PDF inventory: found {total_pdfs} PDFs across {len(inventory_dict['industries'])} industries")
    
    return {
        "status": "success",
        "total_pdfs": total_pdfs,
        "industries": len(inventory_dict["industries"])
    }


def identify_common_reports(**context) -> Dict:
    """
    Identifies common reports within each segment.
    A common report is defined as a PDF that exists across multiple companies in the same segment.
    """
    logging.info("Starting to identify common reports within segments")
    
    # Get inventory from upstream task
    inventory = context['ti'].xcom_pull(key='pdf_inventory', task_ids='list_pdfs_in_s3')
    if not inventory:
        error_msg = "Failed to retrieve PDF inventory from upstream task"
        logging.error(error_msg)
        raise AirflowFailException(error_msg)
    
    # Structure to hold common reports
    common_reports = {
        "timestamp": datetime.now().isoformat(),
        "industries": {}
    }
    
    # Process each industry and segment
    for industry, segments in inventory["industries"].items():
        common_reports["industries"][industry] = {}
        
        for segment, companies in segments.items():
            # Track filenames and their occurrences across companies
            filename_occurrences = defaultdict(set)
            
            # Count occurrences of each filename across companies
            for company, files in companies.items():
                for filename in files:
                    filename_occurrences[filename].add(company)
            
            # Consider a report "common" if it appears in at least 2 companies
            common_files = {
                filename: list(companies)
                for filename, companies in filename_occurrences.items()
                if len(companies) >= 2
            }
            
            if common_files:
                common_reports["industries"][industry][segment] = common_files
    
    # Store results in XCom for downstream tasks
    context['ti'].xcom_push(key='common_reports', value=common_reports)
    
    # Count total common reports for logging
    total_common = sum(
        len(segments) 
        for industry in common_reports["industries"].values() 
        for segments in industry.values()
    )
    
    logging.info(f"Identified {total_common} common reports across all segments")
    
    return {
        "status": "success",
        "total_common_reports": total_common
    }

def generate_inventory_report(**context) -> Dict:
    """
    Generates and stores comprehensive inventory report in S3.
    """
    logging.info("Generating comprehensive inventory report")
    
    # Get data from upstream tasks
    inventory = context['ti'].xcom_pull(key='pdf_inventory', task_ids='list_pdfs_in_s3')
    common_reports = context['ti'].xcom_pull(key='common_reports', task_ids='identify_common_reports')
    total_pdfs = context['ti'].xcom_pull(key='total_pdfs', task_ids='list_pdfs_in_s3')
    
    if not inventory or not common_reports:
        error_msg = "Failed to retrieve inventory or common reports from upstream tasks"
        logging.error(error_msg)
        raise AirflowFailException(error_msg)
    
    # Create comprehensive report
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_pdfs": total_pdfs,
            "industries": len(inventory["industries"]),
            "segments": sum(len(segments) for segments in inventory["industries"].values()),
            "companies": sum(
                len(companies) 
                for industry in inventory["industries"].values() 
                for companies in industry.values()
            )
        },
        "inventory": inventory,
        "common_reports": common_reports
    }
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Create temp file for the report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(report, temp_file, indent=2, default=str)
        temp_path = temp_file.name
    
    # Upload to S3
    s3_key = f"{OUTPUT_PREFIX}/inventory_report_{timestamp}.json"
    
    s3_hook.load_file(
        filename=temp_path,
        key=s3_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    # Clean up temp file
    os.unlink(temp_path)
    
    logging.info(f"Inventory report saved to s3://{S3_BUCKET}/{s3_key}")
    
    return {
        "status": "success",
        "report_path": f"s3://{S3_BUCKET}/{s3_key}",
        "summary": report["summary"]
    }

# Create DAG
dag = DAG(
    "s3_pdf_inventory_pipeline",
    default_args=default_args,
    description="Lists PDFs in S3 organized by industry/segment/company and identifies common reports",
    schedule_interval="@weekly",  # Run weekly, adjust as needed
    catchup=False
)

# Define tasks
list_pdfs_task = PythonOperator(
    task_id='list_pdfs_in_s3',
    python_callable=list_pdfs_in_s3,
    dag=dag
)

identify_common_reports_task = PythonOperator(
    task_id='identify_common_reports',
    python_callable=identify_common_reports,
    dag=dag
)

generate_report_task = PythonOperator(
    task_id='generate_inventory_report',
    python_callable=generate_inventory_report,
    dag=dag
)

# Set task dependencies
list_pdfs_task >> identify_common_reports_task >> generate_report_task 