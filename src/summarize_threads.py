from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from utils import llm
from rag_prep import load_index
from prompts import Summarization_map_prompt, Summarization_refine_prompt
import json
import time

def summarize_tweets():

    with open("../data/reddit_threads_data.txt", "r", encoding="utf-8") as tweet_data:
        data = json.load(tweet_data)

    map_prompt = PromptTemplate(template=Summarization_map_prompt, input_variables=["text"])
    # combine_prompt = PromptTemplate(template=Summarization_refine_prompt, input_variables=["text"])

    # map_reduce_chain = load_summarize_chain(
    #     llm,
    #     chain_type="refine",
    #     map_prompt=map_prompt,
    #     # prompt=map_prompt,
    #     combine_prompt=combine_prompt,
    #     # return_intermediate_steps=True,
    # )

    chain = map_prompt | llm | StrOutputParser()

    summary_list = []
    thread_summary_data = []
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=300)
    for item in data:
        title = item.split('$$')[0]
        item = item.split('$$')[1]
        print(f"\nSummarizing Reddit Thread Tokens: ",len(item.split()))
        split_docs = text_splitter.create_documents([item])
        summary = ""
        for doc in split_docs:
            summary+=chain.invoke(doc)
            time.sleep(2)

        # chain.batch(split_docs)
        summary_list.append(summary)
        thread_summary_data.append({'title': title, 'summary': summary[:1000]})
    
    with open("../data/summary.txt", "w", encoding="utf-8") as summary_file:
        json.dump(summary_list, summary_file)
    print("\nSummarization Completed")
    load_index()

    return thread_summary_data