#!/bin/python3

import argparse
import logging
from dotenv import load_dotenv, find_dotenv
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate

from lib.gpu_util import check_gpu
from lib.document_store import DocumentStore
from lib.prompt import RAG_PROMPT
from lib.taide_chat import TaideChatModel

logger = logging.getLogger(__name__)

def retrieval_augmented_generation(db_path:str, question:str):
    
    store = DocumentStore.load(db_path)
    retriever = store.as_retriever()

    llm = TaideChatModel()
    llm.load_model()
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | RAG_PROMPT 
        | llm
    )
    result = rag_chain.invoke(question)
    print(result.content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QA with local documents and RAG.')
    parser.add_argument("db_path", help="the path to the vector database.", type=str)
    parser.add_argument("question", help="your question.", type=str)
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

    retrieval_augmented_generation(args.db_path, args.question)