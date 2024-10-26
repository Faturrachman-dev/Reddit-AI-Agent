from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
# from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from praw import Reddit
import requests
import os
import json
import tweepy
import json
import gradio as gr
import re

# RAG
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# llm = ChatGroq(model="llama3-70b-8192")
llm = ChatGroq(model="llama3-8b-8192")
loader = TextLoader('tweet_data.txt')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print("Len of RAG Docs: ",len(docs))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# docs_text = [doc.page_content for doc in docs]
# doc_embeddings = embeddings.embed_documents(docs_text)
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("RAG_index")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

RAG_PROMPT = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
     
Question: {query} 

Context: {context} 

Answer:
"""
prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["query","context"])

def ask_questions(query,history):

    results = vectorstore.similarity_search(query,k=5)
    print("RAG Retrived: ", results)
    context = format_docs(results)
    chain = prompt | llm | StrOutputParser()
    results = chain.invoke({'query':query,'context':context})

    return results