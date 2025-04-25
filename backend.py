import os
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

# Configuration parameters
PDF_DIRECTORY = "resources/pdf_documents"      # Directory containing your PDF documents
CHROMA_DB_DIRECTORY = "resources/chroma_db"    # Directory to store ChromaDB
CHUNK_SIZE = 1000                              # Text chunk size for splitting documents
CHUNK_OVERLAP = 200                            # Overlap between chunks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_PATH = "meta-llama/Llama-3.2-1B"    # Base model
PEFT_MODEL_PATH = "mental_health_chat_llm"    # PEFT adapter path
MAX_NEW_TOKENS = 256                           # Max new tokens to generate

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store model and vector store
tokenizer = None
model = None
vector_store = None
rag_pipeline = None

def load_documents():
    """Load PDF documents from directory"""
    print("Loading PDF documents...")
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        print(f"Created directory {PDF_DIRECTORY}")
        return []
    
    loader = DirectoryLoader(PDF_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    """Split documents into chunks for better retrieval"""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks=None):
    """Create or load ChromaDB vector store"""
    print("Setting up vector store...")
    
    # Using a lightweight embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )
    
    # Create directory if it doesn't exist
    if not os.path.exists(CHROMA_DB_DIRECTORY):
        os.makedirs(CHROMA_DB_DIRECTORY, exist_ok=True)
    
    # Create or load the vector store
    if os.path.exists(os.path.join(CHROMA_DB_DIRECTORY, "chroma.sqlite3")) and chunks is None:
        print("Loading existing ChromaDB...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedding_model
        )
    elif chunks:
        print("Creating new ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_DIRECTORY
        )
        vector_store.persist()
    else:
        # Create an empty vector store if no chunks provided and no existing DB
        print("Creating empty ChromaDB...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedding_model
        )
        vector_store.persist()
    
    print("Vector store setup complete")
    return vector_store

def load_peft_model():
    """Load the PEFT fine-tuned Llama model"""
    print("Loading PEFT fine-tuned model...")
    
    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    # Enable padding on the right side
    tokenizer.padding_side = "right"
    
    # Add special tokens if they don't exist
    special_tokens = {
        "pad_token": "<PAD>",
        "eos_token": "</s>",
        "bos_token": "<s>"
    }
    
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            tokenizer.add_special_tokens({token_type: token})
    
    # Load base model
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load PEFT adapter on top of base model
    print(f"Loading PEFT adapter from {PEFT_MODEL_PATH}...")
    model = PeftModel.from_pretrained(
        base_model,
        PEFT_MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"PEFT model loaded successfully (device: {DEVICE})")
    return tokenizer, model

def setup_rag_pipeline(vector_store, tokenizer, model):
    """Set up the RAG pipeline"""
    print("Setting up RAG pipeline...")
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    
    # Create a text generation pipeline
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
        device_map="auto"
    )
    
    def query_rag_system(query, chat_history=None):
        """Process a user query using the RAG system"""
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Extract document metadata for response
        doc_sources = []
        for doc in retrieved_docs:
            if 'source' in doc.metadata:
                source = os.path.basename(doc.metadata['source'])
                if source not in doc_sources:
                    doc_sources.append(source)
        
        # Format the context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prepare the prompt with retrieved context
        if chat_history and len(chat_history) > 0:
            # Format chat history
            history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
            prompt = f"""Context information from mental health resources:
{context}

Previous conversation:
{history_text}

User: {query}
Assistant:"""
        else:
            prompt = f"""Context information from mental health resources:
{context}

User: {query}
Assistant:"""
        
        # Generate response
        response = generation_pipeline(
            prompt, 
            return_full_text=False,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )[0]["generated_text"]
        
        return {
            "response": response.strip(),
            "sources": doc_sources,
            "retrieved_docs_count": len(retrieved_docs)
        }
    
    print("RAG pipeline setup complete")
    return query_rag_system

def initialize_system():
    """Initialize the entire system"""
    global tokenizer, model, vector_store, rag_pipeline
    
    # Load documents and create vector store if not already initialized
    if vector_store is None:
        documents = load_documents()
        chunks = split_documents(documents) if documents else None
        vector_store = create_vector_store(chunks)
    
    # Load model if not already initialized
    if tokenizer is None or model is None:
        tokenizer, model = load_peft_model()
    
    # Set up RAG pipeline if not already initialized
    if rag_pipeline is None:
        rag_pipeline = setup_rag_pipeline(vector_store, tokenizer, model)
    
    return "System initialized successfully"

# API routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "Mental health counselor API is running"})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint to process user queries"""
    # Initialize system if not already initialized
    if rag_pipeline is None:
        initialize_system()
    
    # Get request data
    data = request.json
    query = data.get('query')
    chat_history = data.get('chat_history', [])
    
    # Validate input
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Process query through RAG pipeline
    try:
        result = rag_pipeline(query, chat_history)
        return jsonify({
            "response": result["response"],
            "sources": result["sources"],
            "retrieved_docs_count": result["retrieved_docs_count"]
        })
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_pdf', methods=['POST'])
def upload_pdf():
    """Endpoint to upload a new PDF document to the system"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Ensure directory exists
        if not os.path.exists(PDF_DIRECTORY):
            os.makedirs(PDF_DIRECTORY, exist_ok=True)
        
        # Save file
        file_path = os.path.join(PDF_DIRECTORY, file.filename)
        file.save(file_path)
        
        # Reload documents and update vector store
        try:
            # Load the new PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to vector store
            global vector_store
            if vector_store is None:
                # Initialize vector store if not already done
                vector_store = create_vector_store()
            
            # Add documents to existing vector store
            vector_store.add_documents(chunks)
            vector_store.persist()
            
            return jsonify({
                "message": "File uploaded and indexed successfully",
                "filename": file.filename,
                "chunks_added": len(chunks)
            })
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Only PDF files are allowed"}), 400

@app.route('/api/reset_chat', methods=['POST'])
def reset_chat():
    """Endpoint to reset the chat history"""
    return jsonify({"message": "Chat history reset successfully"})

if __name__ == '__main__':
    # Initialize the system on startup
    initialize_system()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 1234))
    app.run(host='0.0.0.0', port=port, debug=False)