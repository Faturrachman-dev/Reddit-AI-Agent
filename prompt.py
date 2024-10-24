# prompt = ChatPromptTemplate.from_messages([
    # ('system', ''' 
    # Task: Summarize the Twitter thread comments.

    # Input:
    # Thread Name: [Name of the Twitter thread]
    # Comments: [Text containing all the comments from different users within the thread]
    
    # Output:
    # A concise summary of the key points and discussions raised in the Twitter thread comments.
    # Example:
    # Thread Name: "The Future of AI"

    # Comments:
    # User1: "AI has immense potential to revolutionize industries, but we must address ethical concerns."
    # User2: "I'm excited about AI's ability to solve complex problems, but I worry about job displacement."
    # User3: "We need to prioritize AI safety and ensure it's developed responsibly."
    
    # Summary:
    # The Twitter thread on "The Future of AI" discussed the potential benefits and risks of AI development. Participants expressed excitement about AI's potential to solve complex problems but also raised concerns about ethical implications and job displacement. There was a strong emphasis on the need for responsible AI development and prioritizing safety.

    # using above example and prompt create summary of below data
    # ===Comments===
    # {data}
    # ''')
    # ])