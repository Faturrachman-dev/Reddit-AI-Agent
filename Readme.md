# 🌟 **Reddit AI Agent**  
**Your Smart Reddit Assistant!**  
Reddit AI Agent is an intelligent tool that helps you explore Reddit like never before! 🔎 It allows you to search for any query and fetch top Reddit threads along with their most relevant comments. 🚀  

✨ **Core Features:**  
- 🔥 **Retrieve Top Threads**: Find the most relevant threads for any query.  
- 📝 **Summarize Content**: Get concise summaries of thread discussions and comments.  
- 💬 **Conversational Interface**: Chat with an AI-powered bot to get personalized answers.  
- 🎨 **User-Friendly UI**: Enjoy a sleek design with built-in logging and monitoring.  

![Demo Image](python_programming.png)

---

## 🛠️ **Features**  
- **🔗 Fetch Top Reddit Threads**: Quickly find top discussions based on your query.  
- **📖 Summarize Content**: Saves you time by summarizing threads and their comments.  
- **🤖 Conversational Chat**: Chat with summarized content for a personalized experience.  
- **✨ Easy-to-Use Interface**: Smooth, intuitive design for a great user experience.  

---

## 📚 **Tools and Libraries**  
- 🐍 **PRAW**: For fetching Reddit data (threads, comments).  
- 🔗 **LangChain & LangSmith**: Build and manage AI chains and logs.  
- 🤖 **Groq API**: Supports natural language processing with an LLM.  
- 📦 **FAISS Vector Store**: Efficiently index and search data.  
- 🌟 **Hugging Face Embeddings**: Convert text into embeddings for semantic searches.  

---

## 🔑 **API Keys Needed**  
1. **🔐 Reddit API Key**: [How to obtain](https://www.geeksforgeeks.org/how-to-get-client_id-and-client_secret-for-python-reddit-api-registration/).  
2. **🔐 Groq API Key**: Register at [Groq Console](https://console.groq.com/).  

---

## ⚙️ **Setup and Run**  
1. **🛠️ Create `.env` file** with the following keys:  
   ```env
   REDDIT_API_CLIENT_ID="<your_reddit_client_id>"
   REDDIT_API_SECRET="<your_reddit_secret>"
   REDDIT_USER_AGENT="<your_user_agent>"
   GROQ_API_KEY="<your_groq_key>"
   ```
2. 📦 **Installation and Running:**
   ```env
   pip install -r requirements.txt
   cd src
   gradio UI.py
   ```

---

## 📝 **TODO**
- 📥 Add more data sources: Twitter, Quora.
- ☁️ Add support for cloud embeddings using Pinecone.
- 🌐 Deploy to cloud platforms.
