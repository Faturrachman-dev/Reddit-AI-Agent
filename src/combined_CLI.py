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
from datetime import datetime

# RAG
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from prompts import Summarization_map_prompt, Summarization_refine_prompt, RAG_prompt

# NEW: Add imports for Google models and TokenTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import TokenTextSplitter

load_dotenv()

def get_reddits(topic, n):
    reddit = Reddit(
        client_id=os.environ.get("REDDIT_API_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_API_SECRET"),
        user_agent=os.environ.get("REDDIT_USER_AGENT"),
    )

    threads = []
    for submission in reddit.subreddit("all").search(topic, sort="relevance", limit=n):
        print(f"[{datetime.now()}] Reddit Threads Title: {submission.title}")
        submission.comments.replace_more(limit=0)

        # --- MODIFIED:  Get nested comments and engagement ---
        comments = ""
        for top_level_comment in submission.comments:
            # Check if the comment is valid (not deleted, not removed)
            if top_level_comment and not top_level_comment.banned_by:
                comments += f"[{top_level_comment.score}] {top_level_comment.body}\n"  # Include score

                # Get replies (nested comments)
                for reply in top_level_comment.replies:
                    if reply and not reply.banned_by:
                        comments += f"  [{reply.score}] {reply.body}\n"  # Indent replies
        # --- END MODIFIED ---

        threads.append({
            "title": submission.title,
            "comments": comments,
        })

    return threads


def summarize_threads(data):
    # NEW: Use Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    map_prompt = PromptTemplate(template=Summarization_map_prompt, input_variables=["text"])
    combine_prompt = PromptTemplate(template=Summarization_refine_prompt, input_variables=["text"])

    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
    )

    res = []
    # NEW: Use TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)  # Adjust as needed

    for thread in data:
        print(f"[{datetime.now()}] Summarizing thread: {thread['title']}")
        split_docs = text_splitter.split_text(thread['comments'])  # Split text, not Documents
        split_docs = [Document(page_content=t) for t in split_docs] # Convert to List of Document
        summary = map_reduce_chain.invoke(split_docs)
        res.append(summary["output_text"])

    return res


def ask_questions():
    # NEW: Use Gemini 1.5 Flash and Google embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    loader = TextLoader('../data/summary.txt')
    documents = loader.load()
    # NEW: Use TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100) # Adjust as needed
    docs = text_splitter.split_documents(documents)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(template=RAG_prompt, input_variables=["query", "context"])

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit', 'stop', 'break']:
            break

        # Create vectorstore on-the-fly for each query (for simplicity)
        vectorstore = FAISS.from_documents(docs, embeddings)  # Create vectorstore here
        results = vectorstore.similarity_search(query, k=3)
        context = format_docs(results)
        chain = prompt | llm | StrOutputParser()
        results = chain.invoke({'query': query, 'context': context})
        print(f"[{datetime.now()}] AI: {results}")


def main():
    topic = input("Please enter topic you want to know about!! ")
    n = int(input("Enter no of tweets you want to capture "))

    print(f"[{datetime.now()}] Fetching Reddit Threads please wait...")
    threads = get_reddits(topic, n)

    os.makedirs("../data", exist_ok=True)

    with open("../data/reddit_threads_data.json", "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=4)
    print(f"[{datetime.now()}] Reddit threads saved to data/reddit_threads_data.json")

    print(f"[{datetime.now()}] Please wait we are summarizing all the Reddit Threads...")
    summary = summarize_threads(threads)
    print('=' * 30)
    print(f"[{datetime.now()}] Here is the summary of threads: ", summary)

    with open("../data/summary.txt", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=4)
    print(f"[{datetime.now()}] Summary saved to data/summary.txt")

    user_prompt = input('Do you want to ask questions on summary: ')
    if user_prompt.lower() == 'yes':
        print(f"[{datetime.now()}] Please wait we are processing summary so we can perform RAG on it...")
        ask_questions()
    else:
        print('Thanks for your time, GoodBye!!')


if __name__ == "__main__":
    main()