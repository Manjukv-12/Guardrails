import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.chains import RetrievalQA

# Groq API Key Configuration
GROQ_API_KEY = "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"

def load_pdf(file_path):
    """
    Load PDF using PyPDFLoader for better PDF parsing
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Successfully loaded PDF: {file_path}")
        print(f"Total pages loaded: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

# Vector Database Configuration
VECTOR_DB_NAME = "doc_vc"  # Specific name for the vector database

# Load documents
documents = load_pdf('llm-ebook.pdf')  # Replace with your document path

# Check if documents were loaded
if not documents:
    print("No documents to process. Exiting.")
    exit()

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

print(f"Total text chunks created: {len(texts)}")

# Correctly instantiate Hugging Face Embeddings
try:
    # Use a more widely supported embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Lightweight, widely compatible model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    print(f"Error loading embeddings: {e}")
    exit()

# Create vector store with a specific persistent directory
vector_db_path = os.path.join(".", VECTOR_DB_NAME)

try:
    # Ensure the directory exists
    os.makedirs(vector_db_path, exist_ok=True)

    # Create vector database
    vectorstore = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=vector_db_path
    )

    # Verify vector database creation
    print(f"Vector Database '{VECTOR_DB_NAME}' created successfully!")
    print(f"Persistent directory: {vector_db_path}")

    # Create a retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model="llama3-8b-8192",
            temperature=0.2,
            groq_api_key=GROQ_API_KEY
        ),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"  # Most straightforward chain type
    )

    # Function to ask questions about the PDF
    def ask_pdf_question(question):
        try:
            # Retrieve and answer the question
            response = qa_chain.invoke(question)
            return response['result']
        except Exception as e:
            return f"An error occurred: {e}"

    # Example usage
    print("\n--- PDF Question Answering Demo ---")

    # Interactive question-asking loop
    while True:
        user_question = input("\nAsk a question about the PDF (or type 'exit' to quit): ")

        if user_question.lower() == 'exit':
            break

        if user_question:
            answer = ask_pdf_question(user_question)
            print("\nAnswer:", answer)

except Exception as e:
    print(f"Error creating vector database: {e}")
    exit()


# In your main.py, add:
from nemoguardrails.rails import LLMRails
import os

# Initialize guardrails
guardrails = LLMRails.from_path("./guardrails")

# Modify your ask_pdf_question function
def ask_pdf_question(question):
    try:
        # Apply input guardrails
        guardrail_response = guardrails.generate(messages=[{"role": "user", "content": question}])
        
        # If input was flagged as unsafe, return the guardrail response
        if "I cannot provide information" in guardrail_response:
            return guardrail_response
            
        # Otherwise proceed with RAG
        context = vectorstore.similarity_search(question, k=3)
        
        # Check context safety
        context_texts = [doc.page_content for doc in context]
        
        # Generate response with context
        response = qa_chain.invoke(question)
        
        # Apply output guardrails to the response
        safe_response = guardrails.generate(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": response['result']}
            ]
        )
        
        return safe_response
        
    except Exception as e:
        return f"An error occurred: {e}"
    
#####################

