#!/bin/python3

import argparse
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader

from lib.gpu_util import check_gpu
from lib.textract_loader import TextractLoader
from lib.trafilatura_loader import TrafilaturaLoader
from lib.parallel_splitter import ParallelSplitter
from lib.document_store import DocumentStore

logger = logging.getLogger(__name__)

def construct_db(docs_path: str, output_path: str, force_html: bool):
    """
    Construct vector database from local documents and save to the destination.
    """

    loader = DirectoryLoader(docs_path,
                         recursive=True,
                         loader_cls=TextractLoader if not force_html else TrafilaturaLoader,
                         use_multithreading=True,
                         show_progress=True)
    logger.info(f'Loading documents...')
    docs = loader.load()
    logger.debug(docs)
    logger.info(f'Loaded {len(docs)} documents.')

    logger.info(f'Chunking documents...')
    splitter = ParallelSplitter(chunk_size=512, chunk_overlap=128)
    chunks = splitter.split(docs)
    logger.info(f'Chunked documents into {len(chunks)} chunks.')

    db = DocumentStore()
    logger.info(f'Constructing vector store...')
    db.from_documents(chunks)
    logger.info(f'Vector store constructed.')
    db.save(output_path)
    logger.info(f'Saved vector store to {output_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct a FAISS vector database from local documents.')
    parser.add_argument("docs_path", help="the path to the directory of input documents.", type=str)
    parser.add_argument("output_path", help="the path where the final database will be stored.", type=str)
    parser.add_argument("--force-html", help="Force parse the files as HTML.", action="store_true")
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
    
    construct_db(args.docs_path, args.output_path, args.force_html)