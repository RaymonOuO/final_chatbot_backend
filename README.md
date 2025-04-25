## Setup

```
# Create a new conda environment named 'rag'
!conda create -n rag python=3.9 -y

# Activate the environment
!conda activate rag

# Install PyTorch with CUDA support (adjust cuda version if needed)
!conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install basic dependencies
!pip install transformers accelerate bitsandbytes sentencepiece fastapi uvicorn pydantic peft flask flask_cors

# Install LangChain, ChromaDB and other RAG-related packages
!pip install langchain chromadb sentence-transformers pypdf langchain-community


# Install PDF processing libraries
!pip install pdfminer.six pymupdf

# Install optional but useful packages
!pip install tqdm ipywidgets jupyterlab
```

## How to run

```
python backend.py
```
