from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from utils import llm
from prompts import RAG_prompt

prompt = ""
vectorstore = ""
def load_index():
    global prompt
    global vectorstore

    loader = TextLoader('../data/reddit_threads_data.txt')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print("\nLoaded RAG Documents: ",len(docs))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local("../index/RAG_index")
    prompt = PromptTemplate(template=RAG_prompt, input_variables=["query","context"])
    print("\nIndex Created and Saved Successfully")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_questions(query, history):
    
    results = vectorstore.similarity_search(query,k=5)
    print("\nQuery: ",query)
    print("\nRAG Retrived Documents: ")
    for doc in results:
        print(doc)
        print('-'*90)
    context = format_docs(results)
    chain = prompt | llm | StrOutputParser()
    results = chain.invoke({'query':query,'context':context})
    print("\nAI: ",results)

    return results