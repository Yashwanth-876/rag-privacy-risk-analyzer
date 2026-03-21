 # Privacy Policy & Terms Analyzer — RAG Application

 An AI powered privacy risk analyzer built using Retrival Agumented Generation (RAG).
 Upload any terms & conditions txt files and instantly detect potential risk and feedback using semantic search and Google Gemini
 Also Deployed on Streamlit Cloud 
 [Privacy Risk Analyzer](https://rag-privacy-analyzer-yash.streamlit.app/)

 The Application follows two step architecture
 **Data Ingestion Phase**
 Upload T&C Text → Text Extraction → Text Chunking → Sentence Transformer Embeddings → Store in ChromaDB
 **Query Phase**
 User Question → Embed Question → ChromaDB Retrieval → Prompt Construction → Google Gemini → Risk Analysis
