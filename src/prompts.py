Summarization_map_prompt = """
    You are provided with reddit thread comments. Most of the comments are discussions related to problems and solutions users are facing. 
    Create a summary of all the solutions, suggestions and key points mentioned in the discussions and dont return in markdown format it should contain only letters no special characters no new lines direct return summary in a paragraph. 
    ===Comments===
    {text}
"""

Summarization_refine_prompt = """
    The following is a set of summaries:
    {text}
    Take these and distill it into a final, consolidated summary of the main themes and dont return in markdown format it should contain only letters no special characters no new lines direct return summary in a paragraph.
"""

RAG_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Question: {query} 
    Context: {context} 
    Answer:
"""