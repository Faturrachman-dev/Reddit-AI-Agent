import gradio as gr
import re
import json
from combined_CLI import get_reddits, summarize_threads, ask_questions
from langchain_core.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import TokenTextSplitter

CSS ="""
#component-0 > div:nth-child(11) { height: 600px !important; }
"""

def display_text_list():
    try:
        with open("../data/reddit_threads_data.json", "r", encoding="utf-8") as f:
            threads = json.load(f)
        text_list = summarize_threads(threads)
    except FileNotFoundError:
        return "Error: Reddit threads data not found.  Fetch threads first."
    except Exception as e:
        return f"Error summarizing: {e}"

    formatted_data=""
    for i, item in enumerate(text_list):
        title = f"<h3>Summary {i+1}</h3>"
        new_sum = re.sub(r'\n', ' ', item)
        summary = f"<p>{new_sum}</p>"
        formatted_data+=title
        formatted_data+=summary
    return formatted_data

def fetch_and_summarize(topic, num_threads):
    try:
        num_threads = int(num_threads)
        threads = get_reddits(topic, num_threads)

        os.makedirs("../data", exist_ok=True)
        with open("../data/reddit_threads_data.json", "w", encoding="utf-8") as f:
            json.dump(threads, f, indent=4)

        summaries = summarize_threads(threads)

        with open("../data/summary.txt", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4)

        raw_snippet = json.dumps(threads[:1], indent=4)
        return raw_snippet, "\n\n".join(summaries)

    except Exception as e:
        return f"Error: {e}", ""

def answer_question(question):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ.get("GOOGLE_API_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ.get("GOOGLE_API_KEY"))

    try:
        with open("../data/summary.txt", "r", encoding="utf-8") as f:
            summaries = json.load(f)
    except FileNotFoundError:
        return "Error: Summary file not found. Please fetch and summarize threads first."
    except Exception as e:
        return f"Error loading summaries: {e}"

    combined_summary = "\n\n".join(summaries)

    docs = [Document(page_content=combined_summary)]

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(template=RAG_prompt, input_variables=["query", "context"])

    vectorstore = FAISS.from_documents(docs, embeddings)
    results = vectorstore.similarity_search(question, k=3)
    context = format_docs(results)
    chain = prompt | llm | StrOutputParser()
    results = chain.invoke({'query': question, 'context': context})
    return results

with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# AI Reddit Summarizer")

    gr.Markdown("<br />")
    gr.ChatInterface(fn=answer_question, type="messages", examples=["hello", "hola", "merhaba"], title="Chat with Reddit Threads")

    with gr.Tab("Fetch and Summarize"):
        topic_input = gr.Textbox(label="Topic")
        num_threads_input = gr.Textbox(label="Number of Threads")
        fetch_button = gr.Button("Fetch and Summarize")
        raw_output = gr.Textbox(label="Raw Threads (Snippet)")
        summary_output = gr.Textbox(label="Summaries")

        fetch_button.click(
            fetch_and_summarize,
            inputs=[topic_input, num_threads_input],
            outputs=[raw_output, summary_output]
        )

    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Question")
        answer_button = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer")

        answer_button.click(
            answer_question,
            inputs=[question_input],
            outputs=[answer_output]
        )

demo.launch()