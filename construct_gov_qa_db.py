#!/bin/python3

import argparse
import logging
import csv
import jinja2
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import DirectoryLoader

from lib.gpu_util import check_gpu
from lib.textract_loader import TextractLoader
from lib.parallel_splitter import ParallelSplitter
from lib.document_store import DocumentStore

logger = logging.getLogger(__name__)

doc_template = jinja2.Environment(loader=jinja2.BaseLoader).from_string(
"""
類似問題：{{prompt}}
{% if resource != '' %}來自{{resource}}的答案：{% else %}參考答案：{% endif %}
{{response}}
{% if resource != '' %}參考資料：{{resource}}常見問答
{% endif %}{% if postd_date != '' %}日期：{{postd_date}}{% endif %}
"""
)

index_template = list(
    map(
        jinja2.Environment(loader=jinja2.BaseLoader).from_string,
        ["{{prompt}}", "{{response}}"]
    )
)

def construct_db(dataset_file: str, output_path: str):
    """
    Construct vector database from preprocessed QA dataset and save to the destination.
    """

    dataset = []
    documents = []
    embeddings = []

    logger.info(f'Loading documents...')
    with open(dataset_file, newline='') as f:
        reader = csv.DictReader(f)
        dataset = list(reader)
    documents = [doc_template.render(**qa).strip() for qa in dataset for _ in index_template]
    logger.info(f'Loaded {len(dataset)} documents.')

    db = DocumentStore()
    logger.info(f'Calculating embeddings...')
    indexes = [t.render(**qa) for qa in dataset for t in index_template]
    embeddings = db.embedding_model.embed_documents(indexes)
    logger.info(f'Embedding calculated.')

    logger.info(f'Constructing vector store...')
    text_embedding_pairs = list(zip(documents, embeddings))
    db.from_embeddings(text_embedding_pairs)
    logger.info(f'Vector store constructed.')
    db.save(output_path)
    logger.info(f'Saved vector store to {output_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct a FAISS vector database from local documents.')
    parser.add_argument("dataset_file", help="the file of the preprocessed QA dataset (CSV format).", type=str)
    parser.add_argument("output_path", help="the path where the final database will be stored.", type=str)
    parser.add_argument("--log", help="the log level. (INFO, DEBUG, ...)", type=str, default="INFO")
    args = parser.parse_args()
    
    # Setup logger
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log}')
    logging.basicConfig(level=numeric_level)
    
    # Read local .env file
    load_dotenv(find_dotenv())
    check_gpu()
    
    construct_db(args.dataset_file, args.output_path)