"""
Industry Report Processing Pipeline: PDF → Markdown → Vectors → Pinecone
Features comprehensive debugging, analytics, and error recovery mechanisms.
Preserves images and tables in markdown output.
"""
import json
import os
import time
import uuid
import tempfile
import shutil
import traceback
import re
import logging
import psutil
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(override=True)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.base import BaseHook
from airflow.exceptions import AirflowFailException
from airflow.models import Variable, TaskInstance

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'industry.json')
with open(config_path, 'r') as f:
    config = json.load(f)

AWS_CONN_ID = config["AWS_CONN_ID"]
S3_BUCKET = config["S3_BUCKET"]
S3_BASE_FOLDER = "healthcare-industry"
S3_Chunk_BASE_FOLDER = "healthcare_industry/chunks"
PINECONE_INDEX_NAME = config["PINECONE_INDEX_NAME"]

# Default arguments for DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 9),
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}

# =====================================================================================
# MONITORING FRAMEWORK
# =====================================================================================

class ProcessingMetrics:
    """Comprehensive monitoring framework for pipeline telemetry."""
    def __init__(self, report_name: str):
        self.report_name = report_name
        self.start_time = time.time()
        self.stages = {}
        self.page_metrics = {}
        self.chunk_metrics = {}
        self.errors = []
        self.warnings = []
        self.memory_samples = []
        self.sample_memory()
        
    def sample_memory(self) -> float:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.memory_samples.append((time.time(), memory_mb))
        return memory_mb
        
    def start_stage(self, stage_name: str, metadata: Dict = None) -> None:
        self.stages[stage_name] = {
            'start_time': time.time(),
            'status': 'running',
            'memory_before': self.sample_memory(),
            'metadata': metadata or {}
        }
        logging.info(f"📊 Starting stage: {stage_name}")
        
    def end_stage(self, stage_name: str, status: str = 'success', metrics: Dict = None) -> Dict:
        if stage_name not in self.stages:
            self.warning(f"Attempted to end unknown stage: {stage_name}")
            return {}
            
        end_time = time.time()
        memory_after = self.sample_memory()
        duration = end_time - self.stages[stage_name]['start_time']
        memory_change = memory_after - self.stages[stage_name]['memory_before']
        
        self.stages[stage_name].update({
            'end_time': end_time,
            'duration': duration,
            'memory_after': memory_after,
            'memory_change': memory_change,
            'status': status,
            'metrics': metrics or {}
        })
        
        log_method = logging.info if status == 'success' else logging.warning
        log_method(f"📊 Completed stage: {stage_name} ({status}) in {duration:.2f}s with {memory_change:.2f}MB memory change")
        
        return self.stages[stage_name]
    
    def error(self, stage: str, error_msg: str, error_obj: Exception = None, context: Dict = None) -> None:
        error_entry = {
            'timestamp': time.time(),
            'stage': stage,
            'error': error_msg,
            'error_type': type(error_obj).__name__ if error_obj else 'Unknown',
            'context': context or {},
        }
        
        if error_obj:
            error_entry['traceback'] = traceback.format_exception(
                type(error_obj), error_obj, error_obj.__traceback__
            )
        
        self.errors.append(error_entry)
        logging.error(f"❌ ERROR in {stage}: {error_msg}")
    
    def warning(self, message: str, context: Dict = None) -> None:
        self.warnings.append({
            'timestamp': time.time(),
            'message': message,
            'context': context or {}
        })
        logging.warning(f"⚠️ WARNING: {message}")
    
    def get_summary(self) -> Dict:
        duration = time.time() - self.start_time
        memory_values = [m for _, m in self.memory_samples]
        
        return {
            'report_name': self.report_name,
            'duration_seconds': duration,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'completed_stages': [s for s, details in self.stages.items() if details.get('status') == 'success'],
            'failed_stages': [s for s, details in self.stages.items() if details.get('status') == 'failed'],
            'memory_stats': {
                'peak_memory_mb': max(memory_values),
                'initial_memory_mb': self.memory_samples[0][1],
                'final_memory_mb': self.memory_samples[-1][1],
                'memory_change_mb': self.memory_samples[-1][1] - self.memory_samples[0][1]
            }
        }

    def save_report(self, s3_hook: S3Hook, report_name: str) -> str:
        report = {
            'summary': self.get_summary(),
            'stages': self.stages,
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(report, temp_file, indent=2, default=str)
            temp_path = temp_file.name
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"{S3_BASE_FOLDER}/diagnostics/{report_name}_{timestamp}.json"
        
        s3_hook.load_file(
            filename=temp_path,
            key=s3_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        os.unlink(temp_path)
        logging.info(f"📊 Diagnostic report saved to s3://{S3_BUCKET}/{s3_key}")
        return s3_key

# Initialize global metrics tracker
metrics = None  # Will be initialized per report

# =====================================================================================
# PDF PROCESSING FUNCTIONS
# =====================================================================================

def locate_pdf(**context) -> str:
    """Locate the PDF file in S3 and download it locally."""
    global metrics
    
    # Get the report info from DAG run configuration or use defaults
    dag_run = context['dag_run']
    if dag_run and dag_run.conf:
        report_info = dag_run.conf.get('report_info')
    else:
        # Use default values if no configuration is provided
        report_info = {
            "industry": "healthcare-industry",
            "segment": "otc-pharmaceutical-segment",
            "company": "perrigo-plc",
            "filename": "PRGO (Perrigo Company plc)  (10-K) 2025-02-28.pdf"
        }
        logging.info(f"Using default report_info: {report_info}")
    
    # Validate report info (now optional as we provide defaults)
    if not report_info or not isinstance(report_info, dict):
        report_info = {
            "industry": "healthcare-industry",
            "segment": "otc-pharmaceutical-segment",
            "company": "perrigo-plc",
            "filename": "report.pdf"
        }
        logging.info(f"Invalid report_info format, using defaults: {report_info}")
    
    # Store report_info in XCom for other tasks
    context['ti'].xcom_push(key='report_info', value=report_info)
    
    # Initialize metrics for this report
    metrics = ProcessingMetrics(report_info['filename'])
    metrics.start_stage("pdf_location")

    # After initializing metrics
    metrics_dict = {
        "report_name": metrics.report_name,
        "start_time": metrics.start_time,
        "stages": metrics.stages,
        # Add other necessary attributes
    }
    context['ti'].xcom_push(key='metrics_dict', value=metrics_dict)

    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Construct S3 key from report info
    s3_key = f"{report_info['industry']}/{report_info['segment']}/{report_info['company']}/{report_info['filename']}"
    
    if s3_hook.check_for_key(s3_key, bucket_name=S3_BUCKET):
        logging.info(f"✅ PDF found in S3 at s3://{S3_BUCKET}/{s3_key}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="industry_pdf_")
        local_pdf_path = os.path.join(temp_dir, report_info['filename'])
        
        # Download the file
        s3_hook.get_key(s3_key, bucket_name=S3_BUCKET).download_file(local_pdf_path)
        logging.info(f"✅ PDF downloaded to {local_pdf_path}")
        
        # Store temp directory for later cleanup
        context['ti'].xcom_push(key='temp_dir', value=temp_dir)
        context['ti'].xcom_push(key='pdf_path', value=local_pdf_path)
        
        metrics.end_stage("pdf_location", status="success")
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        return local_pdf_path
    else:
        error_msg = f"❌ PDF not found in S3: s3://{S3_BUCKET}/{s3_key}"
        metrics.error("pdf_location", error_msg)
        metrics.end_stage("pdf_location", status="failed")
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        raise AirflowFailException(error_msg)

def process_pdf_with_mistral(**context) -> dict:
    """Convert PDF to markdown using MistralAI, preserving images and tables."""
    # Retrieve and rebuild metrics
    metrics_dict = context['ti'].xcom_pull(key='metrics_dict', task_ids='locate_pdf')
    metrics = ProcessingMetrics(metrics_dict["report_name"])
    metrics.start_time = metrics_dict["start_time"]
    metrics.stages = metrics_dict["stages"]
    # Restore other attributes

    metrics.start_stage("pdf_processing")
    ti = context['ti']
    
    pdf_path = ti.xcom_pull(key='pdf_path', task_ids='locate_pdf')
    if not pdf_path or not os.path.exists(pdf_path):
        error_msg = f"PDF path not found: {pdf_path}"
        metrics.error("pdf_processing", error_msg)
        metrics.end_stage("pdf_processing", status="failed")
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        raise AirflowFailException(error_msg)

    # Create a temporary directory for markdown output
    temp_output_dir = tempfile.mkdtemp(prefix="mistral_output_")
    temp_markdown_file = os.path.join(temp_output_dir, "output.md")
    
    try:
        from utils.mistralparsing_userpdf import process_pdf
        
        start_time = time.time()
        
        # FIXED: Pass output_dir parameter
        logging.info(f"Processing {os.path.basename(pdf_path)} ...")
        process_pdf(pdf_path=Path(pdf_path), output_dir=Path(temp_output_dir))
        
        # Check if file was created
        markdown_path = temp_markdown_file
        
        duration = time.time() - start_time
        
        if os.path.exists(markdown_path):
            char_count = len(Path(markdown_path).read_text(encoding='utf-8'))
        else:
            error_msg = f"Markdown file not created at {markdown_path}"
            metrics.error("pdf_processing", error_msg)
            metrics.end_stage("pdf_processing", status="failed")
            
            updated_metrics_dict = {
                "report_name": metrics.report_name,
                "start_time": metrics.start_time,
                "stages": metrics.stages,
                "errors": metrics.errors,
                "warnings": metrics.warnings,
                "memory_samples": metrics.memory_samples
            }
            context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
            
            raise AirflowFailException(error_msg)
        
        logging.info(f"Mistral PDF parse completed: {markdown_path} (chars={char_count})")
        
        # XCom push path for the next step
        ti.xcom_push(key='mistral_markdown_path', value=markdown_path)
        # Also store the temp directory for cleanup
        ti.xcom_push(key='temp_output_dir', value=temp_output_dir)

        metrics.end_stage("pdf_processing", status="success", metrics={
            "duration": duration,
            "char_count": char_count
        })
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        return {"markdown_path": markdown_path}
    
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        error_msg = f"Error in process_pdf_with_mistral: {e}"
        metrics.error("pdf_processing", error_msg, e)
        metrics.end_stage("pdf_processing", status="failed")
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        raise AirflowFailException(error_msg)

# =====================================================================================
# CHUNKING AND EMBEDDING
# =====================================================================================

class MarkdownChunker:
    """Advanced chunking strategy that preserves markdown structure."""
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text: str) -> List[str]:
        """Split markdown text while preserving structure."""
        chunks = []
        current_chunk = ""
        
        # Split by headers first
        header_splits = self._split_by_headers(text)
        
        for section in header_splits:
            # If section is small enough, keep it as is
            if len(section) <= self.chunk_size:
                if current_chunk and len(current_chunk + "\n\n" + section) <= self.chunk_size:
                    current_chunk += "\n\n" + section
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = section
            else:
                # For larger sections, split by markdown blocks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                block_chunks = self._split_by_blocks(section)
                chunks.extend(block_chunks)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Apply overlap
        if self.chunk_overlap > 0:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                    overlapped_chunks.append(overlap_text + "\n\n" + chunks[i])
            chunks = overlapped_chunks
        
        return chunks
    
    def _split_by_headers(self, text: str) -> List[str]:
        """Split text at markdown headers."""
        import re
        header_pattern = re.compile(r'^#{1,6}\s+[^\n]+$', re.MULTILINE)
        
        # Find all headers
        headers = list(header_pattern.finditer(text))
        if not headers:
            return [text]
        
        sections = []
        start_pos = 0
        
        for match in headers:
            if match.start() > start_pos:
                sections.append(text[start_pos:match.start()].strip())
            start_pos = match.start()
        
        # Add final section
        if start_pos < len(text):
            sections.append(text[start_pos:].strip())
        
        return [s for s in sections if s]  # Remove empty sections
    
    def _split_by_blocks(self, text: str) -> List[str]:
        """Split text by markdown blocks (tables, code blocks, etc.)."""
        chunks = []
        current_chunk = ""
        
        # Split into lines
        lines = text.split('\n')
        in_code_block = False
        in_table = False
        table_content = []
        
        for line in lines:
            # Check for code block markers
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                
                # If entering code block and current chunk is too big
                if in_code_block and len(current_chunk) >= self.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                current_chunk += line + '\n'
                continue
            
            # Check for table markers
            if line.strip().startswith('|'):
                if not in_table:
                    in_table = True
                    # If current chunk is too big, save it
                    if len(current_chunk) >= self.chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                table_content.append(line)
                continue
            elif in_table:
                # Table ended
                in_table = False
                table_text = '\n'.join(table_content) + '\n'
                table_content = []
                
                if len(current_chunk) + len(table_text) > self.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = table_text
                else:
                    current_chunk += table_text
                continue
            
            # Regular text processing
            if in_code_block or in_table:
                current_chunk += line + '\n'
            else:
                if len(current_chunk) + len(line) + 1 > self.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class FastChunker:
    """High-performance chunking strategy optimized for speed rather than structure preservation."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Quickly split text using a sliding window approach."""
        # Handle empty or small text
        if not text or len(text) <= self.chunk_size:
            return [text]
        
        # Preprocess text to identify "hard boundaries" we shouldn't split across
        # (like headers, code blocks, tables) and mark them with special tokens
        lines = text.split('\n')
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        # Fast sliding window approach
        while current_pos < text_length:
            # Calculate end position for this chunk
            end_pos = min(current_pos + self.chunk_size, text_length)
            
            # If we're not at the end, try to find a good split point
            if end_pos < text_length:
                # Look for paragraph breaks first (double newline)
                paragraph_break = text.rfind('\n\n', current_pos, end_pos)
                if paragraph_break != -1 and paragraph_break > current_pos:
                    end_pos = paragraph_break + 2  # Include the double newline
                else:
                    # Fallback to single newline
                    newline = text.rfind('\n', current_pos, end_pos)
                    if newline != -1 and newline > current_pos:
                        end_pos = newline + 1  # Include the newline
                    # If no good break point, just use character boundary
            
            # Extract the chunk
            chunk = text[current_pos:end_pos].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move position for next chunk, accounting for overlap
            current_pos = end_pos - self.chunk_overlap if self.chunk_overlap > 0 else end_pos
        
        return chunks
    
    def split_text_batched(self, text: str, batch_size=10) -> List[str]:
        """Even faster chunking by processing text in batches."""
        # For extremely large documents, process in batches
        if len(text) > self.chunk_size * batch_size * 5:  # Arbitrary threshold
            segments = []
            segment_size = len(text) // batch_size + 1
            
            # Split into rough segments first
            for i in range(0, len(text), segment_size):
                segment = text[i:i+segment_size]
                # Process each segment
                segments.extend(self.split_text(segment))
                
            return segments
        else:
            # For smaller documents, use regular chunking
            return self.split_text(text)

def process_chunks_and_embeddings(**context):
    """Process chunks and create embeddings for all PDFs in all segment folders."""
    # Retrieve and rebuild metrics
    ti = context['ti']
    metrics_dict = ti.xcom_pull(key='metrics_dict', task_ids='locate_pdf')
    metrics = ProcessingMetrics(metrics_dict["report_name"])
    metrics.start_time = metrics_dict["start_time"]
    metrics.stages = metrics_dict["stages"]
    if "errors" in metrics_dict:
        metrics.errors = metrics_dict["errors"]
    if "warnings" in metrics_dict:
        metrics.warnings = metrics_dict["warnings"]
    if "memory_samples" in metrics_dict:
        metrics.memory_samples = metrics_dict["memory_samples"]
    
    metrics.start_stage("chunking_and_embedding")
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Get all segment folders
    segment_folders = s3_hook.list_keys(
        bucket_name=S3_BUCKET,
        prefix=f"{S3_BASE_FOLDER}/",
        delimiter='/'
    )
    
    if not segment_folders:
        error_msg = f"No segment folders found in S3 bucket: {S3_BUCKET}/{S3_BASE_FOLDER}/"
        metrics.error("chunking_and_embedding", error_msg)
        metrics.end_stage("chunking_and_embedding", status="failed")
        raise AirflowFailException(error_msg)
    
    processed_files = 0
    try:
        # Process each segment folder
        for segment_folder in segment_folders:
            segment_name = segment_folder.rstrip('/').split('/')[-1]
            logging.info(f"Processing segment folder: {segment_name}")
            
            # Get all company folders in this segment
            company_folders = s3_hook.list_keys(
                bucket_name=S3_BUCKET,
                prefix=segment_folder,
                delimiter='/'
            )
            
            for company_folder in company_folders:
                company_name = company_folder.rstrip('/').split('/')[-1]
                logging.info(f"Processing company folder: {company_name} in segment: {segment_name}")
                
                # List all files in this company folder
                files = s3_hook.list_keys(
                    bucket_name=S3_BUCKET,
                    prefix=company_folder
                )
                
                pdf_files = [f for f in files if f.lower().endswith('.pdf')]
                
                if not pdf_files:
                    logging.info(f"No PDF files found in company folder: {company_folder}")
                    continue
                
                # Process each PDF file in this company folder
                for pdf_file in pdf_files:
                    try:
                        filename = os.path.basename(pdf_file)
                        logging.info(f"Processing PDF file: {filename}")
                        
                        # Download the PDF file locally
                        temp_dir = tempfile.mkdtemp(prefix="industry_pdf_")
                        local_pdf_path = os.path.join(temp_dir, filename)
                        s3_hook.get_key(pdf_file, bucket_name=S3_BUCKET).download_file(local_pdf_path)
                        
                        # Extract industry from the path
                        path_parts = pdf_file.split('/')
                        industry = path_parts[0] if len(path_parts) > 0 else "unknown"
                        
                        # Process the PDF to markdown
                        temp_output_dir = tempfile.mkdtemp(prefix="mistral_output_")
                        temp_markdown_file = os.path.join(temp_output_dir, "output.md")
                        from utils.mistralparsing_userpdf import process_pdf
                        process_pdf(pdf_path=Path(local_pdf_path), output_dir=Path(temp_output_dir))
                        
                        # Read markdown content
                        with open(temp_markdown_file, "r", encoding="utf-8") as f:
                            markdown_text = f.read()
                        
                        # Use FastChunker for text splitting
                        chunker = FastChunker(chunk_size=500, chunk_overlap=50)
                        chunks = chunker.split_text(markdown_text)
                        # Or for very large documents:
                        # chunks = chunker.split_text_batched(markdown_text)
                        
                        logging.info(f"Created {len(chunks)} chunks from markdown text for file: {filename}")
                        
                        # Prepare report info
                        report_info = {
                            "industry": industry,
                            "segment": segment_name,
                            "company": company_name,
                            "filename": filename
                        }
                        
                        # Store chunks in S3
                        chunks_s3_key = f"{S3_BASE_FOLDER}-chunks/{segment_name}/{company_name}/{filename.replace('.pdf', '')}_chunks.json"
                        
                        chunks_dict = {
                            f"chunk_{i}": {
                                "text": chunk,
                                "length": len(chunk),
                                "metadata": report_info,
                                "timestamp": datetime.now().isoformat()
                            }
                            for i, chunk in enumerate(chunks)
                        }
                        
                        chunks_payload = {
                            "report_name": filename,
                            "created_at": datetime.now().isoformat(),
                            "total_chunks": len(chunks),
                            "metadata": report_info,
                            "chunks": chunks_dict
                        }
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            json.dump(chunks_payload, f, indent=2)
                            chunks_path = f.name
                        
                        s3_hook.load_file(
                            filename=chunks_path,
                            key=chunks_s3_key,
                            bucket_name=S3_BUCKET,
                            replace=True
                        )
                        
                        os.unlink(chunks_path)
                        logging.info(f"Stored {len(chunks)} chunks in S3: s3://{S3_BUCKET}/{chunks_s3_key}")
                        
                        # Initialize BGE embedding model
                        from FlagEmbedding import FlagModel
                        bge_model = FlagModel('BAAI/bge-large-en-v1.5', 
                                            use_fp16=True,
                                            device='cuda' if torch.cuda.is_available() else 'cpu')
                        
                        # Process chunks in batches
                        batch_size = 8
                        embeddings = []
                        
                        for i in range(0, len(chunks), batch_size):
                            batch = chunks[i:i+batch_size]
                            batch_embeddings = bge_model.encode(batch, max_length=512)
                            
                            # Normalize embeddings
                            from sklearn.preprocessing import normalize
                            batch_embeddings = normalize(batch_embeddings)
                            
                            embeddings.extend(batch_embeddings.tolist())
                            
                            if i % 50 == 0:
                                logging.info(f"Embedded chunks {i} to {i+len(batch)} of {len(chunks)}")
                        
                        # Initialize Pinecone
                        from pinecone import Pinecone, ServerlessSpec
                        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                        
                        # Create index if it doesn't exist
                        index_name = PINECONE_INDEX_NAME
                        if index_name not in pc.list_indexes().names():
                            pc.create_index(
                                name=index_name,
                                dimension=1024,
                                metric="cosine",
                                spec=ServerlessSpec(
                                    cloud="aws",
                                    region="us-east-1"
                                )
                            )
                            logging.info(f"Created Pinecone index '{index_name}'")
                        
                        # Get index
                        index = pc.Index(index_name)
                        
                        # Use segment as namespace
                        namespace = f"{industry}-{segment_name}"
                        
                        # Prepare vectors with metadata
                        vectors = []
                        for i, emb in enumerate(embeddings):
                            vectors.append({
                                "id": f"{filename}_{i}",
                                "values": emb,
                                "metadata": {
                                    "chunk_id": f"chunk_{i}",
                                    "industry": industry,
                                    "segment": segment_name,
                                    "company": company_name,
                                    "filename": filename,
                                    "s3_chunks_key": chunks_s3_key,
                                    "embedding_model": "bge-large-en-v1.5",
                                    "processing_timestamp": datetime.now().isoformat()
                                }
                            })
                        
                        # Upsert vectors in batches
                        batch_size = 100
                        for i in range(0, len(vectors), batch_size):
                            batch = vectors[i:i+batch_size]
                            index.upsert(vectors=batch, namespace=namespace)
                            logging.info(f"Upserted vectors {i} to {i+len(batch)} of {len(vectors)}")
                        
                        processed_files += 1
                        logging.info(f"Successfully processed file {pdf_file}")
                        
                        # Clean up temporary files
                        shutil.rmtree(temp_dir)
                        shutil.rmtree(temp_output_dir)
                    
                    except Exception as e:
                        logging.error(f"Error processing file {pdf_file}: {str(e)}")
                        metrics.error("processing_pdf", f"Error processing {pdf_file}: {str(e)}", e)
                        # Continue to next file rather than failing the whole task
                        continue
        
        metrics.end_stage("chunking_and_embedding", status="success", metrics={
            "processed_files": processed_files,
        })
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        ti.xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        return {
            "status": "success",
            "processed_files": processed_files
        }
        
    except Exception as e:
        error_msg = f"Error in process_chunks_and_embeddings: {str(e)}"
        metrics.error("chunking_and_embedding", error_msg, e)
        metrics.end_stage("chunking_and_embedding", status="failed")
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        ti.xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        raise AirflowFailException(error_msg)

# =====================================================================================
# CLEANUP AND REPORTING
# =====================================================================================

def cleanup_and_report(**context):
    """Clean up temporary files and generate processing report."""
    # Retrieve and rebuild metrics
    ti = context['ti']
    metrics_dict = ti.xcom_pull(key='metrics_dict', task_ids='locate_pdf')
    metrics = ProcessingMetrics(metrics_dict["report_name"])
    metrics.start_time = metrics_dict["start_time"]
    metrics.stages = metrics_dict["stages"]
    # Restore other necessary attributes
    
    metrics.start_stage("cleanup")
    
    try:
        # Clean up temp directories
        temp_dir = context['ti'].xcom_pull(key='temp_dir', task_ids='locate_pdf')
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")
            
        # Clean up temporary output directory
        temp_output_dir = context['ti'].xcom_pull(key='temp_output_dir', task_ids='process_pdf_with_mistral')
        if temp_output_dir and os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            logging.info(f"Cleaned up temporary output directory: {temp_output_dir}")
        
        # Generate and save processing report
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        report_key = metrics.save_report(s3_hook, "processing_report")
        
        metrics.end_stage("cleanup", status="success")
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        return {
            "status": "success",
            "report_path": f"s3://{S3_BUCKET}/{report_key}"
        }
        
    except Exception as e:
        error_msg = f"Error in cleanup_and_report: {str(e)}"
        metrics.error("cleanup", error_msg, e)
        metrics.end_stage("cleanup", status="failed")
        
        updated_metrics_dict = {
            "report_name": metrics.report_name,
            "start_time": metrics.start_time,
            "stages": metrics.stages,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "memory_samples": metrics.memory_samples
        }
        context['ti'].xcom_push(key='metrics_dict', value=updated_metrics_dict)
        
        raise AirflowFailException(error_msg)

# =====================================================================================
# DAG DEFINITION
# =====================================================================================

dag = DAG(
    "industry_report_to_vector_pipeline",
    default_args=default_args,
    description="Process industry reports to vectors with advanced chunking and BGE embeddings",
    schedule_interval=None,  # Manual trigger
    catchup=False,
    doc_md="""
    # Industry Report Processing Pipeline
    
    This DAG processes industry reports from PDF to vectors for semantic search.
    
    ## Required Configuration
    
    When triggering this DAG, you must provide the following configuration in JSON format:
    
    ```json
    {
        "report_info": {
            "industry": "healthcare-industry",
            "segment": "otc-pharmaceutical-segment",
            "company": "perrigo-plc",
            "filename": "report.pdf"
        }
    }
    ```
    
    ### Fields Description:
    - industry: The industry folder name in S3
    - segment: The segment folder name
    - company: The company folder name
    - filename: The PDF file name
    
    All folder names should use hyphens (-) instead of spaces.
    """
)

locate_pdf_task = PythonOperator(
    task_id='locate_pdf',
    python_callable=locate_pdf,
    dag=dag
)

process_pdf_task = PythonOperator(
    task_id='process_pdf_with_mistral',
    python_callable=process_pdf_with_mistral,
    dag=dag
)

chunking_task = PythonOperator(
    task_id='process_chunks_and_embeddings',
    python_callable=process_chunks_and_embeddings,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_and_report',
    python_callable=cleanup_and_report,
    dag=dag
)

# Set task dependencies
locate_pdf_task >> process_pdf_task >> chunking_task >> cleanup_task