#  "cells": [
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "# RAG Chatbot with PDF Knowledge Base (Optimized for Apple Silicon)\n",
#     "\n",
#     "This notebook demonstrates how to build a high-performance Retrieval-Augmented Generation (RAG) chatbot optimized to run on Apple Silicon (M-series chips). \n",
#     "\n",
#     "We will use:\n",
#     "- **Generator LLM:** `unsloth/llama-3-8b-Instruct-bnb-4bit` (A powerful, quantized version of Llama 3)\n",
#     "- **Retriever Model:** `BAAI/bge-large-en-v1.5` (A top-tier embedding model)\n",
#     "- **Vector Store:** ChromaDB"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "### 1. Install Dependencies\n",
#     "\n",
#     "First, we need to install the necessary Python libraries. `accelerate` and `bitsandbytes` are required to load the quantized 4-bit model efficiently."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "!pip install transformers torch sentence-transformers pypdf chromadb datasets accelerate bitsandbytes -q"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "### 2. Import Libraries and Check for Apple Silicon GPU\n",
#     "\n",
#     "We'll import all the required libraries and verify that PyTorch can see the Mac's `mps` device (the GPU)."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import torch\n",
#     "from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
#     "from sentence_transformers import SentenceTransformer\n",
#     "import chromadb\n",
#     "from pypdf import PdfReader\n",
#     "from datasets import Dataset\n",
#     "import numpy as np\n",
#     "import textwrap\n",
#     "\n",
#     "# Check for Apple Silicon GPU and set the device\n",
#     "if torch.backends.mps.is_available():\n",
#     "    device = torch.device(\"mps\")\n",
#     "    print(\"MPS (Apple Silicon GPU) is available. Using device: mps\")\n",
#     "else:\n",
#     "    device = torch.device(\"cpu\")\n",
#     "    print(\"MPS not available. Using device: cpu\")"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "### 3. Load and Process the PDF\n",
#     "\n",
#     "This step remains the same. We'll load the `Aluminium.pdf` file, extract its text content, and then split the text into smaller, manageable chunks."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "def extract_text_from_pdf(pdf_path):\n",
#     "    \"\"\"Extracts text from a PDF file.\"\"\"\n",
#     "    reader = PdfReader(pdf_path)\n",
#     "    text = \"\"\n",
#     "    for page in reader.pages:\n",
#     "        text += page.extract_text() or \"\"\n",
#     "    return text\n",
#     "\n",
#     "def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):\n",
#     "    \"\"\"Splits text into overlapping chunks.\"\"\"\n",
#     "    chunks = []\n",
#     "    current_pos = 0\n",
#     "    while current_pos < len(text):\n",
#     "        end_pos = current_pos + chunk_size\n",
#     "        chunk = text[current_pos:end_pos]\n",
#     "        chunks.append(chunk)\n",
#     "        current_pos += chunk_size - chunk_overlap\n",
#     "    return [chunk for chunk in chunks if chunk.strip()] # Remove empty chunks\n",
#     "\n",
#     "# Specify the path to your PDF file\n",
#     "pdf_path = 'Aluminium.pdf'\n",
#     "\n",
#     "# Extract and chunk the text\n",
#     "pdf_text = extract_text_from_pdf(pdf_path)\n",
#     "text_chunks = split_text_into_chunks(pdf_text)\n",
#     "\n",
#     "# Create a Hugging Face Dataset\n",
#     "documents_dict = {'text': text_chunks}\n",
#     "dataset = Dataset.from_dict(documents_dict)\n",
#     "\n",
#     "print(f\"Successfully loaded and split the PDF into {len(dataset)} chunks.\")"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "### 4. Create Text Embeddings with BGE-Large\n",
#     "\n",
#     "We'll use the new, more accurate BGE embedding model to convert our text chunks into numerical vectors. This will run on your Mac's GPU automatically."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "embedding_model_name = 'BAAI/bge-large-en-v1.5'\n",
#     "embedding_model = SentenceTransformer(embedding_model_name, device=device)\n",
#     "\n",
#     "# Generate embeddings for each chunk\n",
#     "embeddings = embedding_model.encode(dataset['text'], show_progress_bar=True)\n",
#     "\n",
#     "# Add the embeddings to our dataset\n",
#     "dataset = dataset.add_column('embeddings', embeddings.tolist())\n",
#     "\n",
#     "print(\"Embeddings created with BGE-Large and added to the dataset.\")"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "### 5. Build the ChromaDB Collection\n",
#     "\n",
#     "This step remains the same. We will load the documents and their new embeddings into our in-memory vector store."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# Create a ChromaDB client (this will be an in-memory instance)\n",
#     "client = chromadb.Client()\n",
#     "\n",
#     "# Create a new collection or get it if it already exists\n",
#     "collection = client.get_or_create_collection(name=\"aluminium_kb_v2\")\n",
#     "\n",
#     "doc_ids = [str(i) for i in range(len(dataset))]\n",
#     "documents_list = [doc for doc in dataset['text']]\n",
#     "\n",
#     "# Add the documents and their embeddings to the collection\n",
#     "collection.add(\n",
#     "    embeddings=np.array(dataset['embeddings']),\n",
#     "    documents=documents_list,\n",
#     "    ids=doc_ids\n",
#     ")\n",
#     "\n",
#     "print(f\"ChromaDB collection created with {collection.count()} documents.\")"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "### 6. Define the RAG Chatbot with Llama 3\n",
#     "\n",
#     "This is the core of our new chatbot. We load the 4-bit quantized Llama 3 model and create a pipeline that runs on the Mac's GPU (`mps`).\n",
#     "\n",
#     "**Note:** The first time you run this cell, it will download the Llama 3 model, which is several gigabytes. This may take some time."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "llm_model_name = 'unsloth/llama-3-8b-Instruct-bnb-4bit'\n",
#     "\n",
#     "# Configure quantization to load the model in 4-bit\n",
#     "quantization_config = BitsAndBytesConfig(load_in_4bit=True)\n",
#     "\n",
#     "tokenizer = AutoTokenizer.from_pretrained(llm_model_name)\n",
#     "model = AutoModelForCausalLM.from_pretrained(\n",
#     "    llm_model_name,\n",
#     "    quantization_config=quantization_config,\n",
#     "    torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on M-series\n",
#     "    device_map=device # Explicitly set the device\n",
#     ")\n",
#     "\n",
#     "# Create the pipeline for text generation\n",
#     "llm_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)\n",
#     "\n",
#     "def retrieve_context(query, k=3):\n",
#     "    query_embedding = embedding_model.encode([query]).tolist()\n",
#     "    results = collection.query(query_embeddings=query_embedding, n_results=k)\n",
#     "    retrieved_chunks = results['documents'][0]\n",
#     "    return \" \".join(retrieved_chunks)\n",
#     "\n",
#     "def generate_answer(query, context):\n",
#     "    # Llama 3 uses a specific chat template\n",
#     "    messages = [\n",
#     "        {\"role\": \"system\", \"content\": \"You are a helpful assistant. Answer the user's question based on the provided context.\"},\n",
#     "        {\"role\": \"user\", \"content\": f\"Context:\\n{context}\\n\\nQuestion: {query}\"}\n",
#     "    ]\n",
#     "    \n",
#     "    prompt = llm_pipeline.tokenizer.apply_chat_template(\n",
#     "        messages, \n",
#     "        tokenize=False, \n",
#     "        add_generation_prompt=True\n",
#     "    )\n",
#     "\n",
#     "    terminators = [\n",
#     "        llm_pipeline.tokenizer.eos_token_id,\n",
#     "        llm_pipeline.tokenizer.convert_tokens_to_ids(\"<|eot_id|>\" # End of turn token for Llama 3\n",
#     "    ]\n",
#     "\n",
#     "    outputs = llm_pipeline(\n",
#     "        prompt,\n",
#     "        max_new_tokens=256,\n",
#     "        eos_token_id=terminators,\n",
#     "        do_sample=True,\n",
#     "        temperature=0.6,\n",
#     "        top_p=0.9,\n",
#     "    )\n",
#     "    \n",
#     "    # Extract the response from the generated text\n",
#     "    generated_text = outputs[0]['generated_text']\n",
#     "    response = generated_text[len(prompt):].strip()\n",
#     "    return response\n",
#     "\n",
#     "def chatbot(query):\n",
#     "    print(f\"❓ Query: {query}\")\n",
#     "    context = retrieve_context(query)\n",
#     "    answer = generate_answer(query, context)\n",
#     "    print(f\"\\n🤖 Llama 3 Answer:\\n{textwrap.fill(answer, width=80)}\")"
#    ]
#   },
#   {
#    "cell_type": "markdown",
#    "metadata": {},
#    "source": [
#     "### 7. Ask a Question!\n",
#     "\n",
#     "Now, let's test our new high-performance RAG chatbot. The answers should be significantly more detailed, coherent, and human-like."
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "# Example usage\n",
#     "user_query = \"What is red mud and how is it managed in India?\"\n",
#     "chatbot(user_query)"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "user_query_2 = \"Explain the concept of a circular economy for metals like aluminium and copper.\"\n",
#     "chatbot(user_query_2)"
#    ]
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "Python 3",
#    "language": "python",
#    "name": "python3"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.10.9"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 4
# }

