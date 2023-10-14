from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

RAG_PROMPT_TEMPLATE = """請以以下內容為基礎，回答問題。

{% for doc in context %}{{doc.page_content}}

{% endfor %}

問題： {{question}}
答案："""

RAG_PROMPT = PromptTemplate.from_template(
    input_type={'context': [Document], 'question':str},
    template=RAG_PROMPT_TEMPLATE,
    template_format='jinja2'
)