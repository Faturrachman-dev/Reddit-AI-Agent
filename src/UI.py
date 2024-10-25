import gradio as gr
import fetch_threads as core
import summarize_threads as core2

def echo(message, history):
    return "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."

CSS ="""
#component-92 { height: 600px !important; }
"""

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
    gr.Markdown("<h1>Reddit Summarized Threads</h1>")
    textboxes = gr.Column()

    def update_textboxes():
        textboxes = core2.display_text_list()
        return gr.update(components=textboxes)
    
    summarize_tweet_btn.click(core2.display_text_list, inputs=[], outputs=textboxes)

    gr.Markdown("### Tweet 1 Lorem Ipsum")
    gr.Markdown("---")
    gr.Markdown("<b>Summary</b> - Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
    gr.Markdown("### Tweet 2 Lorem Ipsum")
    gr.Markdown("---")
    gr.Markdown("<b>Summary</b> - Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
    gr.Markdown("### Tweet 3 Lorem Ipsum")
    gr.Markdown("---")
    gr.Markdown("<b>Summary</b> - Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
    gr.Markdown("\n\n\n\n\n\n")
    gr.Markdown("\n\n\n\n\n\n")
    
    gr.ChatInterface(fn=echo, type="messages", examples=["hello", "hola", "merhaba"], title="Reddit Bot")

demo.launch()