DataBase Question Answering (DBQA)
===
DBQA is an application that utilizes Retrieval-Augmented Generation (RAG) to answer the question related to local documents.

## Feature

- Support various document format. e.g. txt, html, word, pdf, epub, ...

## Dependency Installation
### For Ubuntu

```bash
apt-get install python-dev-is-python3 libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext
pip install textract
```

### For Archlinux

```bash
pacman -S python libxml2 libxslt antiword unrtf poppler pstotext
pip install textract
```

## Usage

1. Construct the vector database from local documents
    ```bash
    python construct_vector_db.py /path/to/documents /path/to/output/directory
    ```