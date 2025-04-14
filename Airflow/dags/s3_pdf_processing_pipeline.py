"""
S3 PDF Processing Pipeline

Processes PDFs in the S3 bucket organized by industry/segment/company:
1. Inventories all PDFs in the specified structure
2. Extracts metadata from the PDFs
3. Identifies common reports across companies within segments
4. Prepares PDFs for vector embedding

This pipeline is designed to work with the existing book_to_vector_pipeline for
downstream vector embedding and storage in Pinecone.
"""
import os
import json
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from pathlib import Path

# For PDF metadata extraction
import PyPDF2
import re

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.exceptions import AirflowFailException
from airflow.models import Variable, TaskInstance

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)

# Configuration - Load from Airflow Variables or use defaults
try:
    config = Variable.get("pdf_processing_config", deserialize_json=True)
except:
    # Default configuration
    config = {
        "AWS_CONN_ID": "aws_default",
        "S3_BUCKET": "finalproject-product",
        "OUTPUT_PREFIX": "processed_pdfs",
        "INDUSTRIES": ["healthcare industry"]
    }

AWS_CONN_ID = config.get("AWS_CONN_ID")
S3_BUCKET = config.get("S3_BUCKET", "finalproject-product")
OUTPUT_PREFIX = config.get("OUTPUT_PREFIX", "processed_pdfs")
INDUSTRIES = config.get("INDUSTRIES", ["healthcare industry"])

# Default arguments for DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 9),
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}

def extract_path_components(s3_key: str) -> Dict[str, str]:
    """
    Extract industry, segment, company, and filename from S3 key.
    
    Expected format: industry/segment/company/filename.pdf
    """
    components = s3_key.split('/')
    
    if len(components) < 4 or not components[-1].lower().endswith('.pdf'):
        return None
    
    return {
        "industry": components[0],
        "segment": components[1],
        "company": components[2],
        "filename": components[3]
    }

def list_pdfs_in_s3(**context) -> Dict:
    """
    Lists all PDFs in the S3 bucket organized by industry/segment/company.
    """
    logging.info(f"Starting to list PDFs in S3 bucket: {S3_BUCKET}")
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Structure to hold our inventory
    inventory = {
        "timestamp": datetime.now().isoformat(),
        "bucket": S3_BUCKET,
        "industries": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    
    # Track total counts
    total_pdfs = 0
    
    # List to store all PDF keys for downstream tasks
    all_pdf_keys = []
    
    # Iterate through each industry
    for industry in INDUSTRIES:
        logging.info(f"Listing PDFs for industry: {industry}")
        
        # List all objects with this industry prefix
        try:
            industry_keys = s3_hook.list_keys(
                bucket_name=S3_BUCKET,
                prefix=f"{industry}/",
                delimiter="/"
            )
            
            if not industry_keys:
                logging.warning(f"No objects found for industry: {industry}")
                continue
                
            # Process each key to extract components
            for key in industry_keys:
                if not key.lower().endswith('.pdf'):
                    continue
                    
                components = extract_path_components(key)
                if not components:
                    continue
                    
                # Add to our inventory structure
                inventory["industries"][components["industry"]][components["segment"]][components["company"]].append(components["filename"])
                all_pdf_keys.append(key)
                total_pdfs += 1
                
                # Log periodically to show progress
                if total_pdfs % 100 == 0:
                    logging.info(f"Processed {total_pdfs} PDFs so far...")
                    
        except Exception as e:
            logging.error(f"Error listing objects for industry {industry}: {str(e)}")
    
    # Convert defaultdict to regular dict for JSON serialization
    inventory_dict = {
        "timestamp": inventory["timestamp"],
        "bucket": inventory["bucket"],
        "industries": {}
    }
    
    for industry, segments in inventory["industries"].items():
        inventory_dict["industries"][industry] = {}
        for segment, companies in segments.items():
            inventory_dict["industries"][industry][segment] = {}
            for company, files in companies.items():
                inventory_dict["industries"][industry][segment][company] = files
    
    # Store results in XCom for downstream tasks
    context['ti'].xcom_push(key='pdf_inventory', value=inventory_dict)
    context['ti'].xcom_push(key='total_pdfs', value=total_pdfs)
    context['ti'].xcom_push(key='all_pdf_keys', value=all_pdf_keys)
    
    logging.info(f"Completed PDF inventory: found {total_pdfs} PDFs across {len(inventory_dict['industries'])} industries")
    
    return {
        "status": "success",
        "total_pdfs": total_pdfs,
        "industries": len(inventory_dict["industries"])
    }

def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extracts metadata from a PDF file.
    
    Returns a dictionary with:
    - Title
    - Author
    - Creation date
    - Modification date
    - Page count
    - First page text (for content preview)
    """
    metadata = {
        "page_count": 0,
        "title": None,
        "author": None,
        "creation_date": None,
        "modification_date": None,
        "first_page_preview": None,
    }
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            # Basic metadata
            metadata["page_count"] = len(pdf_reader.pages)
            
            # Get PDF info if available
            if pdf_reader.metadata:
                metadata["title"] = pdf_reader.metadata.title
                metadata["author"] = pdf_reader.metadata.author
                metadata["creation_date"] = pdf_reader.metadata.creation_date
                metadata["modification_date"] = pdf_reader.metadata.modification_date
            
            # Extract text from first page for preview (limit to 500 chars)
            if pdf_reader.pages and len(pdf_reader.pages) > 0:
                first_page_text = pdf_reader.pages[0].extract_text()
                if first_page_text:
                    # Clean up the text (remove excessive whitespace)
                    first_page_text = re.sub(r'\s+', ' ', first_page_text).strip()
                    metadata["first_page_preview"] = first_page_text[:500] + "..." if len(first_page_text) > 500 else first_page_text
    
    except Exception as e:
        logging.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
        metadata["error"] = str(e)
    
    return metadata

def process_pdfs_metadata(**context) -> Dict:
    """
    Processes a sample of PDFs to extract metadata.
    
    For large inventories, processes a representative sample from each segment.
    """
    logging.info("Starting PDF metadata extraction")
    
    # Get PDF keys from upstream task
    all_pdf_keys = context['ti'].xcom_pull(key='all_pdf_keys', task_ids='list_pdfs_in_s3')
    if not all_pdf_keys:
        error_msg = "No PDF keys found from upstream task"
        logging.error(error_msg)
        raise AirflowFailException(error_msg)
    
    # Get inventory structure
    inventory = context['ti'].xcom_pull(key='pdf_inventory', task_ids='list_pdfs_in_s3')
    if not inventory:
        error_msg = "PDF inventory not found from upstream task"
        logging.error(error_msg)
        raise AirflowFailException(error_msg)
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Create temporary directory for PDF downloads
    temp_dir = tempfile.mkdtemp(prefix="pdf_metadata_")
    context['ti'].xcom_push(key='temp_dir', value=temp_dir)
    
    # Track processed PDFs and their metadata
    metadata_results = {
        "timestamp": datetime.now().isoformat(),
        "industries": defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    }
    
    MAX_PDFS_PER_SEGMENT = 5  # Sample size per segment
    total_processed = 0
    
    try:
        # Process each industry and segment in the inventory
        for industry, segments in inventory["industries"].items():
            for segment, companies in segments.items():
                logging.info(f"Processing segment: {industry}/{segment}")
                
                # Track PDFs processed in this segment
                segment_processed = 0
                
                # Iterate through companies in this segment
                for company, files in companies.items():
                    # Process at most 2 PDFs per company
                    company_processed = 0
                    
                    for filename in files:
                        if segment_processed >= MAX_PDFS_PER_SEGMENT:
                            break
                            
                        if company_processed >= 2:
                            break
                            
                        # Construct S3 key
                        s3_key = f"{industry}/{segment}/{company}/{filename}"
                        
                        # Download the PDF
                        local_path = os.path.join(temp_dir, filename)
                        try:
                            s3_hook.get_key(s3_key, bucket_name=S3_BUCKET).download_file(local_path)
                            
                            # Extract metadata
                            metadata = extract_pdf_metadata(local_path)
                            
                            # Store metadata with file path information
                            metadata["s3_path"] = s3_key
                            metadata["industry"] = industry
                            metadata["segment"] = segment
                            metadata["company"] = company
                            
                            # Add to results
                            metadata_results["industries"][industry][segment][company][filename] = metadata
                            
                            company_processed += 1
                            segment_processed += 1
                            total_processed += 1
                            
                            # Clean up downloaded file
                            os.remove(local_path)
                            
                        except Exception as e:
                            logging.error(f"Error processing {s3_key}: {str(e)}")
                
                logging.info(f"Processed {segment_processed} PDFs for segment {industry}/{segment}")
        
        # Convert defaultdict to regular dict for JSON serialization
        metadata_dict = {
            "timestamp": metadata_results["timestamp"],
            "industries": {}
        }
        
        for industry, segments in metadata_results["industries"].items():
            metadata_dict["industries"][industry] = {}
            for segment, companies in segments.items():
                metadata_dict["industries"][industry][segment] = {}
                for company, files in companies.items():
                    metadata_dict["industries"][industry][segment][company] = files
        
        # Store results in XCom for downstream tasks
        context['ti'].xcom_push(key='pdf_metadata', value=metadata_dict)
        
        logging.info(f"Completed metadata extraction for {total_processed} PDFs")
        
        return {
            "status": "success",
            "processed_pdfs": total_processed
        }
        
    except Exception as e:
        error_msg = f"Error in PDF metadata extraction: {str(e)}"
        logging.error(error_msg)
        raise AirflowFailException(error_msg)
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")

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

def prepare_for_embedding(**context) -> Dict:
    """
    Prepares a list of PDFs for vector embedding processing.
    Creates a task list for the vector embedding pipeline to process.
    
    Prioritizes:
    1. Common reports (highest priority)
    2. Representative samples from each segment
    """
    logging.info("Preparing PDF list for vector embedding")
    
    # Get data from upstream tasks
    inventory = context['ti'].xcom_pull(key='pdf_inventory', task_ids='list_pdfs_in_s3')
    common_reports = context['ti'].xcom_pull(key='common_reports', task_ids='identify_common_reports')
    metadata = context['ti'].xcom_pull(key='pdf_metadata', task_ids='process_pdfs_metadata')
    
    if not inventory or not common_reports:
        error_msg = "Failed to retrieve required data from upstream tasks"
        logging.error(error_msg)
        raise AirflowFailException(error_msg)
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Structure to hold embedding tasks
    embedding_tasks = {
        "timestamp": datetime.now().isoformat(),
        "common_reports": [],
        "segment_samples": [],
        "company_samples": []
    }
    
    # First, add common reports as highest priority
    for industry, segments in common_reports["industries"].items():
        for segment, common_files in segments.items():
            for filename, companies in common_files.items():
                # Use the first company that has this file
                company = companies[0]
                
                embedding_tasks["common_reports"].append({
                    "industry": industry,
                    "segment": segment,
                    "company": company,
                    "filename": filename,
                    "s3_key": f"{industry}/{segment}/{company}/{filename}",
                    "priority": "high",
                    "common_to_companies": companies
                })
    
    # Next, add representative samples from each segment
    # (if not already included as common reports)
    common_keys = {task["s3_key"] for task in embedding_tasks["common_reports"]}
    
    for industry, segments in inventory["industries"].items():
        for segment, companies in segments.items():
            # Get a sample of PDFs from this segment (at most 3 per segment)
            segment_samples = []
            for company, files in companies.items():
                if len(segment_samples) >= 3:
                    break
                    
                # Take first file from each company until we have enough samples
                for filename in files:
                    s3_key = f"{industry}/{segment}/{company}/{filename}"
                    if s3_key not in common_keys and s3_key not in {s["s3_key"] for s in segment_samples}:
                        segment_samples.append({
                            "industry": industry,
                            "segment": segment,
                            "company": company,
                            "filename": filename,
                            "s3_key": s3_key,
                            "priority": "medium",
                        })
                        break
            
            embedding_tasks["segment_samples"].extend(segment_samples)
    
    # Lastly, add representative samples from each company
    # (if not already included in common reports or segment samples)
    existing_keys = common_keys.union({task["s3_key"] for task in embedding_tasks["segment_samples"]})
    
    for industry, segments in inventory["industries"].items():
        for segment, companies in segments.items():
            for company, files in companies.items():
                # Skip if no files for this company
                if not files:
                    continue
                    
                # Check if we already have a file from this company
                company_keys = [f"{industry}/{segment}/{company}/{filename}" for filename in files]
                if any(key in existing_keys for key in company_keys):
                    continue
                    
                # Add the first file from this company
                filename = files[0]
                s3_key = f"{industry}/{segment}/{company}/{filename}"
                
                embedding_tasks["company_samples"].append({
                    "industry": industry,
                    "segment": segment,
                    "company": company,
                    "filename": filename,
                    "s3_key": s3_key,
                    "priority": "low",
                })
    
    # Calculate counts for reporting
    total_tasks = (
        len(embedding_tasks["common_reports"]) + 
        len(embedding_tasks["segment_samples"]) + 
        len(embedding_tasks["company_samples"])
    )
    
    logging.info(f"Prepared {total_tasks} PDF embedding tasks: " +
                f"{len(embedding_tasks['common_reports'])} common reports, " +
                f"{len(embedding_tasks['segment_samples'])} segment samples, " +
                f"{len(embedding_tasks['company_samples'])} company samples")
    
    # Save embedding tasks list to S3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(embedding_tasks, temp_file, indent=2, default=str)
        temp_path = temp_file.name
    
    # Upload to S3
    s3_key = f"{OUTPUT_PREFIX}/embedding_tasks_{timestamp}.json"
    s3_hook.load_file(
        filename=temp_path,
        key=s3_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    # Clean up temp file
    os.unlink(temp_path)
    
    # Store XCom reference
    context['ti'].xcom_push(key='embedding_tasks_s3_key', value=s3_key)
    context['ti'].xcom_push(key='embedding_tasks', value=embedding_tasks)
    
    return {
        "status": "success",
        "tasks_count": total_tasks,
        "tasks_s3_path": f"s3://{S3_BUCKET}/{s3_key}"
    }

def generate_report(**context) -> Dict:
    """
    Generates comprehensive report on the PDF processing pipeline.
    """
    logging.info("Generating comprehensive processing report")
    
    # Get data from upstream tasks
    inventory = context['ti'].xcom_pull(key='pdf_inventory', task_ids='list_pdfs_in_s3')
    metadata = context['ti'].xcom_pull(key='pdf_metadata', task_ids='process_pdfs_metadata')
    common_reports = context['ti'].xcom_pull(key='common_reports', task_ids='identify_common_reports')
    embedding_tasks = context['ti'].xcom_pull(key='embedding_tasks', task_ids='prepare_for_embedding')
    
    # Create comprehensive report
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_pdfs": context['ti'].xcom_pull(key='total_pdfs', task_ids='list_pdfs_in_s3'),
            "processed_metadata": len(metadata["industries"]) if metadata else 0,
            "common_reports": sum(
                len(segments) 
                for industry in common_reports["industries"].values() 
                for segments in industry.values()
            ) if common_reports else 0,
            "embedding_tasks": (
                len(embedding_tasks["common_reports"]) + 
                len(embedding_tasks["segment_samples"]) + 
                len(embedding_tasks["company_samples"])
            ) if embedding_tasks else 0
        },
        "industry_breakdown": {},
        "embedding_pipeline_integration": {
            "tasks_file": context['ti'].xcom_pull(key='embedding_tasks_s3_key', task_ids='prepare_for_embedding'),
            "priority_distribution": {
                "high": len(embedding_tasks["common_reports"]) if embedding_tasks else 0,
                "medium": len(embedding_tasks["segment_samples"]) if embedding_tasks else 0,
                "low": len(embedding_tasks["company_samples"]) if embedding_tasks else 0
            }
        }
    }
    
    # Add industry breakdown
    if inventory and "industries" in inventory:
        for industry, segments in inventory["industries"].items():
            report["industry_breakdown"][industry] = {
                "segments": len(segments),
                "companies": sum(len(companies) for companies in segments.values()),
                "pdfs": sum(
                    len(files) 
                    for companies in segments.values() 
                    for files in companies.values()
                )
            }
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Create temp file for the report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(report, temp_file, indent=2, default=str)
        temp_path = temp_file.name
    
    # Upload to S3
    s3_key = f"{OUTPUT_PREFIX}/processing_report_{timestamp}.json"
    s3_hook.load_file(
        filename=temp_path,
        key=s3_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    # Create a more readable markdown report
    md_report = f"""# PDF Processing Pipeline Report

## Summary
- **Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total PDFs**: {report['summary']['total_pdfs']}
- **Common Reports**: {report['summary']['common_reports']}
- **Embedding Tasks**: {report['summary']['embedding_tasks']}

## Industry Breakdown
"""

    for industry, stats in report.get("industry_breakdown", {}).items():
        md_report += f"""
### {industry}
- Segments: {stats['segments']}
- Companies: {stats['companies']}
- PDFs: {stats['pdfs']}
"""

    md_report += f"""
## Embedding Pipeline Integration
- **High Priority Tasks**: {report['embedding_pipeline_integration']['priority_distribution']['high']} (common reports)
- **Medium Priority Tasks**: {report['embedding_pipeline_integration']['priority_distribution']['medium']} (segment samples)
- **Low Priority Tasks**: {report['embedding_pipeline_integration']['priority_distribution']['low']} (company samples)
- **Tasks File**: `{report['embedding_pipeline_integration']['tasks_file']}`

## Next Steps
1. Run the vector embedding pipeline on the generated tasks file
2. Verify vector embeddings in Pinecone
3. Update the list with additional PDFs as needed
"""

    # Save markdown report
    md_key = f"{OUTPUT_PREFIX}/processing_report_{timestamp}.md"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
        temp_file.write(md_report)
        md_path = temp_file.name
    
    s3_hook.load_file(
        filename=md_path,
        key=md_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    # Clean up temp files
    os.unlink(temp_path)
    os.unlink(md_path)
    
    logging.info(f"Reports saved to:\n" +
               f"- JSON: s3://{S3_BUCKET}/{s3_key}\n" +
               f"- Markdown: s3://{S3_BUCKET}/{md_key}")
    
    return {
        "status": "success",
        "report_paths": {
            "json": f"s3://{S3_BUCKET}/{s3_key}",
            "markdown": f"s3://{S3_BUCKET}/{md_key}"
        }
    }

# Create DAG
dag = DAG(
    "s3_pdf_processing_pipeline",
    default_args=default_args,
    description="Processes PDFs in S3 and prepares them for vector embedding",
    schedule_interval="@weekly",  # Run weekly, adjust as needed
    catchup=False
)

# Define tasks
list_pdfs_task = PythonOperator(
    task_id='list_pdfs_in_s3',
    python_callable=list_pdfs_in_s3,
    dag=dag
)

metadata_task = PythonOperator(
    task_id='process_pdfs_metadata',
    python_callable=process_pdfs_metadata,
    dag=dag
)

common_reports_task = PythonOperator(
    task_id='identify_common_reports',
    python_callable=identify_common_reports,
    dag=dag
)

embedding_prep_task = PythonOperator(
    task_id='prepare_for_embedding',
    python_callable=prepare_for_embedding,
    dag=dag
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    dag=dag
)

# Set task dependencies
list_pdfs_task >> [metadata_task, common_reports_task]
[metadata_task, common_reports_task] >> embedding_prep_task >> report_task 