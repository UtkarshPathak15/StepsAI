# Steps AI Assessment

This repository implements a comprehensive information retrieval system for the NVIDIA CUDA documentation website (https://docs.nvidia.com/cuda/).

## Functionalities

- **Web Crawling:** Crawls the provided website up to a specified depth (default 5) to gather relevant information.
  
- **Data Chunking and Preprocessing:** Chunks scraped data based on semantic similarity and preprocesses text using stop word removal, stemming, and tokenization.
  
- **Sentence Embeddings:** Generates sentence embeddings using a pre-trained sentence transformer model.
  
- **Milvus Integration:** Creates a vector database using Milvus to store the generated embeddings and corresponding metadata.
  
- **Hybrid Retrieval and Answering:** Implements a hybrid retrieval approach combining BM25 with BERT/bi-encoder based methods for retrieving relevant documents from the vector database.

- **Question Answering:**  The retrieved and ranked context is used to produce the final answer using the BERT Question Answering model.
- **User Interface:** Provides a Streamlit interface for users to interact with the system and submit queries (app.py).

## Requirements

- **Ubuntu >= 20.04 (x86_64)**

## Libraries

1. **bs4**
   - Used for web scraping to parse HTML and XML documents.
   - Install: `pip install bs4`
   
2. **milvus**
   - A vector database to store and manage large-scale vector data.
   - Install: `pip install milvus pymilvus`
   
3. **pymilvus**
   - The Python SDK for Milvus, used to interact with the Milvus vector database.
   - Install: `pip install pymilvus`
   
4. **transformers**
   - To utilize the BERT model for assessing ranked text and producing relevant answers to queries using it.
   - Install: `pip install transformers`
   
5. **sentence_transformers**
   - Used for transforming sentences into vector representations.
   - Install: `pip install sentence_transformers`
   
6. **rank_bm25**
   - A Python implementation of the Best Matching 25 (BM25) ranking function, used for text retrieval.
   - Install: `pip install rank_bm25`
   
7. **nltk**
   - The Natural Language Toolkit, used for working with human language data (text).
   - Install: `pip install nltk`
   
8. **streamlit**
   - A framework to create web apps for data science and machine learning projects.
   - Install: `pip install streamlit`

## Usage

### Running the System:

1. Install required libraries.
2. Download download all the files in the same directory.
3. Run `streamlit run main.py` to launch the user interface.

### Note: To run ipynb file, directly run all the cells

### File Structure:

- **required_functions.py:** Contains functions for web crawling, data processing, sentence embedding generation, Milvus interaction, and hybrid retrieval.
  
- **app.py:** Entry point for the Streamlit application.
  
- **crawl_data.json** (Optional): Stores scraped data.
  
- **embeddings.npy** (Optional): Stores pre-computed sentence embeddings.

