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
import time

# RAG
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

def summarize_tweets():

    with open("tweet_data.txt", "r", encoding="utf-8") as tweet_data:
        data = json.load(tweet_data)

    # llm = ChatGroq(model="llama3-70b-8192")
    llm = ChatGroq(model="llama3-8b-8192")

    map_prompt_template = """You are provided with twitter thread comments. Most of the comments are discussions related to problems and solutions users are facing. Creat a summary of all the solutions, suggestions and key points mentioned in the discussions and dont return in markdown format it should contain only letters no special characters no new lines direct return summary in a paragraph. \\n ===Comments=== \\n {text}"""
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
    The following is a set of summaries:
    {text}
    Take these and distill it into a final, consolidated summary
    of the main themes and dont return in markdown format it should contain only letters no special characters no new lines direct return summary in a paragraph.
    """
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        # map_prompt=map_prompt,
        prompt=map_prompt,
        # combine_prompt=combine_prompt,
        # return_intermediate_steps=True,
    )

    chain = map_prompt | llm | StrOutputParser()

    res = []
    final = []
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=300)
    for val in data:
        title = val.split('$$')[0]
        val = val.split('$$')[1]
        print(f"Lenght of Reddit Thread",len(val.split()))
        split_docs = text_splitter.create_documents([val])
        # summary = map_reduce_chain.invoke(split_docs)
        summary = ""
        for doc in split_docs:
            summary+=chain.invoke(doc)
            time.sleep(2)

        # chain.batch(split_docs)
        print(type(summary))
        res.append(summary)
        final.append({'title': title, 'summary': summary})
    
    with open("summary.txt", "w", encoding="utf-8") as summary_file:
        json.dump(res, summary_file)

    return final