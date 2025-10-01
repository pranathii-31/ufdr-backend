# import os
# from dotenv import load_dotenv
# from rich.console import Console

# from langchain_community.vectorstores import FAISS
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.agents import initialize_agent, Tool
# from langchain.memory import ConversationBufferMemory
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq

# # Load environment variables (including API keys)
# load_dotenv()

# # Optional: Disable TensorFlow usage in transformers if not needed
# os.environ["USE_TF"] = "0"
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# # Check API key environment variable
# api_key = os.getenv("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file or environment.")

# # Set up conversation memory
# memory = ConversationBufferMemory(return_messages=True)

# # Load embedding model and FAISS vectorstore index
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.load_local("json_faiss_index", embedding_model, allow_dangerous_deserialization=True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


# # Initialize language model with API key
# llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=api_key)


# # Create QA chain and wrap in a Tool
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# qa_tool = Tool(
#     name="QA",
#     func=lambda q: qa_chain.invoke({"query": q})["result"],
#     description="Retrieve relevant information from the dataset."
# )

# # Initialize agent with tools, memory, and LLM
# agent = initialize_agent(
#     tools=[qa_tool],
#     llm=llm,
#     agent="zero-shot-react-description",
#     memory=memory,
#     verbose=True
# )

# # Interactive chat loop
# print("Assistant is ready! Type 'q' to quit.")
# console = Console()

# while True:
#     query = input("You: ")
#     if query.lower() == "q":
#         break

#     result = agent.invoke(query)
#     output = result.get("output", result)

#     console.print("\n[bold cyan]AI Response:[/bold cyan]\n")
#     console.print(output)

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("ufdr_faiss_combined_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

while True:
    query = input("You: ")
    if query.lower() == "q":
        break

    answer = qa_chain.run(query)
    print(f"AI Response:\n{answer}")
