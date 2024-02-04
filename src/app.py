# Note that the libraries are very experimental and may not work as expected

# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv GPT4All chromadb
# https://blog.langchain.dev/langgraph/

# Resources:
# - https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md
# - https://www.youtube.com/watch?v=bupx08ZgSFg

from re import M
from time import sleep
from sqlalchemy import JSON, UUID
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


def get_vectorstore_from_url(url):
  # get tge textin document from the url
  loader = WebBaseLoader(url)
  document = loader.load()

  # split the document into chunks
  text_splitter = RecursiveCharacterTextSplitter()
  document_chunks = text_splitter.split_documents(document)

  #create a vectorstore from the chunks
  # embedding = OpenAIEmbeddings()
  with st.sidebar:
    # st.info(document_chunks)
    st.info("Loading embeddings with url: " + url)
  embedding = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
  
  # with st.sidebar:
  #   st.info(embedding)

  with st.sidebar:
    st.info("Loading ChromaDB...")

  try:
    vector_store = Chroma.from_documents(documents=document_chunks, embedding=embedding)
  except Exception as e:
    st.error(e)
    vector_store = None
  
  return vector_store

def get_context_retriever_chain(vector_store):
  ollama = Ollama(base_url='http://localhost:11434', model="llama2")
  
  retriever = vector_store.as_retriever()

  prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to lookup in order to get information relevant to the conversation")
  ])

  retriever_chain = create_history_aware_retriever(ollama, retriever, prompt)

  return retriever_chain

def get_conversational_rag_chain(retriever_chain):

  ollama = Ollama(base_url='http://localhost:11434', model="llama2")

  prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
  ])

  stuff_documents_chain = create_stuff_documents_chain(ollama, prompt)

  return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
  # create conversation chain
  retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

  # conversation rag chain
  conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

  response = conversation_rag_chain.invoke({
    "chat_history": st.session_state.chat_history,
    "input": user_query
  })

  return response['answer']



# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖", layout="wide")
st.title("Chat with websites")

# sidebar
with st.sidebar:
  st.header("Settings")
  # website_url = "https://blog.langchain.dev/langgraph/"
  # website_url = "https://en.wikipedia.org/wiki/2024_United_States_presidential_election"
  # website_url = "https://sport.hotnews.ro/stiri-alte_sporturi-26876962-handbal-cincilea-esec-pentru-rapid-grupele-ligii-campionilor.htm"
  # website_url = st.text_input("Website URL", website_url)
  website_url = st.text_input("Website URL")

# This it is working only on the first run, should be refactored to accept a new url

if website_url is None or website_url == "":
  st.info("Please enter a website URL")
else:
  # session state
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a chatbot. How can I help you?"),
    ]
  if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(website_url)

  # user input
  user_query = st.chat_input("Type your message here")
  if user_query is not None and user_query != "":
    response = get_response(user_query)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    # test the context loaded in chromadb
    # retrieved_documents = retriever_chain.invoke({
    #   "chat_history": st.session_state.chat_history,
    #   "input": user_query
    # })
    # st.write(retrieved_documents)

  # conversation
  for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
      with st.chat_message("AI"):
        st.write(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("Human"):
        st.write(message.content)