import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from flashrank import Ranker
# The LangChain tools we installed
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
# 1. Load the API key securely
groq_api_key = None

if "GROQ_API_KEY" in st.secrets:
    # For Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    # For your local laptop
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🚨 API Key not found! Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

# 2. Set up the Web Page UI
st.set_page_config(page_title="Ask My Docs", page_icon="📚")
st.title("📚 Ask My Docs (RAG AI)")
st.write("Upload a PDF and ask questions. The AI will only answer using your document!")

# 3. Set up the Groq Brain (LLM)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=groq_api_key)

# 4. Create the Sidebar for PDF Upload
with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.write("---")
    st.write("Built with Streamlit, ChromaDB, and Groq.")

# 5. The Main Logic
if uploaded_file is not None:
    
    if "vectorstore" not in st.session_state:
        with st.spinner("Reading and understanding your PDF... (This takes a few seconds)"):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            
            st.session_state.vectorstore = vectorstore
            st.success("Document processed successfully!")
    # 6. The Chat Interface (With Memory!)
    
    # Create a place in memory to store the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all the previous chat messages on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create the chat input box at the bottom
    if query := st.chat_input("Ask a question about your PDF..."):
        
        # 1. Save the user's question to memory and display it
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Generate the AI's answer
        with st.spinner("Thinking..."):
            # Get the librarian (Retriever) from memory
            # 1. Grab 15 chunks (The Junior Librarian)
            base_retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 15})
            
            # 2. Rerank them and pick the top 3 (The Senior Librarian)
            compressor = FlashrankRerank(top_n=3)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )
            
            # The Prompt Template
            template = """You are a helpful assistant. Answer the question using ONLY the following context. 
            If the answer is not in the context, say "I don't know based on the document." Do not make up information.
            Be precise and detailed in your answer.

            Context:
            {context}

            Question: {query}
            """
            prompt = ChatPromptTemplate.from_template(template)

            # The Pipeline
            rag_chain = (
                {"context": retriever, "query": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Run the pipeline
            answer = rag_chain.invoke(query)
            
            # 3. Save the AI's answer to memory and display it
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

            # 4. Show the sources (which pages it read)
            with st.expander("See the exact pages I read to find this"):
                docs = retriever.invoke(query)
                for doc in docs:
                    page_num = doc.metadata.get("page", "Unknown")
                    st.write(f"**Page {page_num + 1}:**")
                    st.write(doc.page_content)
                    st.write("---")

else:
    st.info("👈 Please upload a PDF in the sidebar to get started.")