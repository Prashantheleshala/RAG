import streamlit as st
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


file1 = open("gemini_key.txt")
key = file1.read()

chat_model = ChatGoogleGenerativeAI(google_api_key=key, 
                                   model="gemini-1.5-pro-latest")
output_parser = StrOutputParser()
loader = PyPDFLoader("2404.07143.pdf")
pages = loader.load_and_split()
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, 
                                               model="models/embedding-001")
# Embed each chunk and load it into the vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
# Persist the database on drive
db.persist()
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 5})


chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take question and Context from user. Answer to the question, if you have the specific information."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

st.title("RAG System on “Leave No Context Behind” Paper")
question = st.text_input("Enter your question:")
if st.button("Answer"):
    response = rag_chain.invoke(question)
    st.write(response)