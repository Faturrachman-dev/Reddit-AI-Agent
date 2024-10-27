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

# RAG
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

def get_reddits(topic,n):
    reddit = Reddit(
        client_id=os.environ.get("REDDIT_API_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_API_SECRET"),
        user_agent=os.environ.get("REDDIT_USER_AGENT"),
    )

    res = []
    print("Reddit Threads Title: ")
    for submission in reddit.subreddit("all").search(query=topic, sort="relevance", limit=n):
        print(submission.title)
        res.append(submission)

    comments = []
    for value in res:
        res=""
        submission = reddit.submission(id=value)
        res = submission.title + "\n"
        submission.comments.replace_more(limit=0)
        # Extract All level comments using .list() function -> we can extract upro 8000 comments but a lot of it is irrevelant Data
        # for comment in submission.comments.list():
        #     res+=comment.body

        # Only extracting first level comment so we have most important data
        for top_level_comment in submission.comments:
            res+=top_level_comment.body
        comments.append(res)

    return comments
    

def summarize_tweets(data):

    llm = ChatGroq(model="llama3-70b-8192")

    map_prompt_template = """You are provided with twitter thread comments. Most of the comments are discussions related to problems and solutions users are facing. Creat a concise summary of all the solutions and suggestions and key points mentioned in the discussions. \\n ===Comments=== \\n {text}"""
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
    The following is a set of summaries:
    {text}
    Take these and distill it into a final, consolidated summary
    of the main themes.
    """
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
    )

    res = []
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2500, chunk_overlap=125)
    for val in data:
        print(f"Lenght of Reddit Thread",len(val.split()))
        split_docs = text_splitter.create_documents([val])
        summary = map_reduce_chain.invoke(split_docs)
        res.append(summary["output_text"])

    # print(res)
    return res

def ask_questions():

    llm = ChatGroq(model="llama3-70b-8192")

    loader = TextLoader('summary.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    docs_text = [doc.page_content for doc in docs]
    doc_embeddings = embeddings.embed_documents(docs_text)
    vectorstore = FAISS.from_documents(docs, embeddings)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    RAG_PROMPT = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
         
    Question: {query} 
    
    Context: {context} 
    
    Answer:
    """
    prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["query","context"])

    while True:
        query = input("You: ")
        if(query.lower() in ['exit','quit','stop','break']):
            break
        results = vectorstore.similarity_search(query,k=3)
        context = format_docs(results)
        chain = prompt | llm | StrOutputParser()
        results = chain.invoke({'query':query,'context':context})
        print(f"AI: ",results)
    
    return

def main():
    topic = input("Please enter topic you want to know about!! ")
    n = int(input("Enter no of tweets you want to capture "))

    print("Fetching Reddits Threads please wait...")
    documents = get_reddits(topic,n)
    print(f"Here are the documents retreived: ",documents)

    print("Please wait we are summarizing all the Redit Threads...")
    summary = summarize_tweets(documents)
    print('='*30)
    print(f"Here is the summary of threads: ",summary)
    # Save summary to a file named "summary.txt"
    with open("summary.txt", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file)
    print("Summary and documents saved to files.")
    
    user_prompt = input('Do you want to ask questions on summary: ')
    if(user_prompt.lower()=='yes'):
        print("Please wait we are processing summary so we can perform RAG on it...")
        ask_questions()
    else:
        print('Thanks for your time, GoodBye!!')


if __name__ == "__main__":
    main()