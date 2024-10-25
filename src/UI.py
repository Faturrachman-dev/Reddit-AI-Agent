import gradio as gr
import fetch_threads as core
import summarize_threads as core2
import rag_prep as core3
import re

CSS ="""
#component-0 > div:nth-child(11) { height: 600px !important; }
"""

def display_text_list(progress=gr.Progress()):
    text_list = core2.summarize_tweets()
    progress(0.05)
    res=""
    for i, item in enumerate(text_list):
        title = f"<h3>{i+1}. {item['title']}</h3>"
        new_sum = re.sub(r'\n', ' ', item['summary'])
        summary = f"<p>{new_sum}</p>"
        res+=title
        res+=summary
    
    print(res)
    return [res, gr.Markdown(visible=True),]


with gr.Blocks(css=CSS) as demo:
    gr.Markdown("<h1><center>Reddit Agent</center></h1>")
    gr.Markdown("<h4><center>Ask anything you want to search reddit for and chat with Reddit Threads to get personalized answers</center></h4>")
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Search Reddit Threads")
            n = gr.Slider(1,10,value=1,step=1,label="Select no of Reddit Threads you want to fetch", interactive=True)
            
            with gr.Row():
                fetch_tweet_btn = gr.Button(value="Fetch Tweets")
                summarize_tweet_btn = gr.Button(value="Summarize Tweets")
        with gr.Column():
            tweet_details = gr.TextArea(label="Results")

    fetch_tweet_btn.click(core.get_reddit, inputs=[query,n], outputs=[tweet_details])

    gr.Markdown("\n\n\n\n\n\n")
    gr.Markdown("\n\n\n\n\n\n")
    head = gr.Markdown("<h1>Reddit Summarized Threads</h1>", visible=False)
    outp = gr.Markdown()
    summarize_tweet_btn.click(display_text_list, inputs=[], outputs=[outp,head])
    gr.Markdown("\n\n\n\n\n\n")
    gr.Markdown("\n\n\n\n\n\n")
    
    gr.ChatInterface(fn=core3.ask_questions, type="messages", examples=["hello", "hola", "merhaba"], title="Chat with Redit Threads")

demo.launch()