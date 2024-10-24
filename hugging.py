import json
from langchain.schema.document import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Replace with your desired model name
model_name = "sentence-transformers/all-mpnet-base-v2"  # Replace with your desired model name
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# with open("summary.txt", "r", encoding="utf-8") as documents_file_safe:
#     documents = json.load(documents_file_safe)
loader = TextLoader('summary.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# comments = []
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# docs = [Document(page_content=x) for x in text_splitter.split_text(documents)]
# comments.extend(docs)

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name=model_name, tokenizer=tokenizer)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
docs_text = [doc.page_content for doc in docs]
doc_embeddings = embeddings.embed_documents(docs_text)
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

results = vectorstore.similarity_search(
    "How metadata filtering works in langchain rag application",
    k=3,
)

for doc in results:
    print(doc.page_content)
    print('='*20)

vectorstore.save_local("first_index")