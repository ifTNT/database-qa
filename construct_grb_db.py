#!/bin/python3

import argparse
import logging
import json
import jinja2
import glob
import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import DirectoryLoader

from lib.gpu_util import check_gpu
from lib.textract_loader import TextractLoader
from lib.parallel_splitter import ParallelSplitter
from lib.document_store import DocumentStore

logger = logging.getLogger(__name__)

doc_template = jinja2.Environment(loader=jinja2.BaseLoader).from_string(
"""
{% if title_zhtw != '' %}計畫中文名稱：{{title_zhtw}}
{% endif %}{% if title_en != '' %}計畫英文名稱：{{title_en}}
{% endif %}{% if abstract_zhtw != '' %}中文摘要：{{abstract_zhtw}}
{% endif %}{% if abstract_en != '' %}英文摘要：{{abstract_en}}
{% endif %}{% if researcher_zhtw != '' %}研究人員中文：{{researcher_zhtw}}
{% endif %}{% if researcher_en != '' %}研究人員英文：{{researcher_en}}
{% endif %}{% if host != '' %}計畫主持人：{{host}}
{% endif %}{% if year != '' %}計畫年度：{{year}}
{% endif %}{% if execution_organ != '' %}計畫執行單位：{{execution_organ}}
{% endif %}{% if plan_organ != '' %}主管機關：{{plan_organ}}
{% endif %}{% if domain != '' %}研究領域：{{domain}}
{% endif %}{% if type != '' %}研究屬性：{{type}}
{% endif %}
"""
)

def get_indexes(data:dict):
    # indexes = ['title_zh', 'title_en', 'abstract_zhtw', 'abstract_en', 'keywords_zhtw', 'keywords_en']
    indexes = ['title_zh', 'title_en', 'abstract_zhtw', 'abstract_en']
    indexes = [data.get(k) for k in indexes ]
    indexes = list(filter(None, indexes))

    return indexes

def construct_db(dataset_path: str, output_path: str):
    """
    Construct vector database from preprocessed QA dataset and save to the destination.
    """

    dataset = []
    documents = []
    embeddings = []

    logger.info(f'Loading documents...')
    for file_path in glob.glob(f'{dataset_path}/*.jsonl'):
        file_path = os.path.abspath(file_path)
        with open(file_path, 'r') as f:
            dataset += [json.loads(l) for l in f]
    documents = [doc_template.render(**record).strip() for record in dataset for _ in get_indexes(record)]
    logger.info(f'Loaded {len(dataset)} documents.')

    db = DocumentStore()
    logger.info(f'Calculating embeddings...')
    indexes = [idx for record in dataset for idx in get_indexes(record)]
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
    parser.add_argument("dataset_path", help="the path to the directory of the dataset (JSONL format).", type=str)
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
    
    construct_db(args.dataset_path, args.output_path)