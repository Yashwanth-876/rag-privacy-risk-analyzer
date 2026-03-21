## Web Page using Streamlit 
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

# Page config
st.set_page_config(
    page_title="Privacy Policy Risk Analyzer Using RAG",
    layout="centered"
)
# Title
st.markdown(
    "### Privacy Policy & Terms Analyzer</h3>",
    unsafe_allow_html=True
)
st.caption("Upload your Terms & Conditions and detect potential privacy risks!")


# Load environment variables
load_dotenv()
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.getenv("GOOGLE_API_KEY")

# Load Gemini client
client = genai.Client(api_key=api_key)

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load ChromaDB
@st.cache_resource
def load_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="/tmp/chroma_db")
    return chroma_client.get_or_create_collection(name="privacy_policy")

embedding_model = load_embedding_model()
collection = load_chroma_collection()

# Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Retrieve relevant chunks from ChromaDB
def retrieve_relevant_chunks(question, text, n_results=5):
    # Re-embed and store if collection is empty
    if collection.count() == 0:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk]
            )

    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    return results['documents'][0]

# Build prompt
def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""
    You are an expert privacy policy and terms & conditions analyzer.
    Analyze the following clauses and identify potential privacy risks.

    Relevant clauses from the policy:
    {context}

    User Question: {question}

    Please provide your analysis in the following format:

    DIRECT ANSWER:
    Answer the user question directly based on the clauses above.

    RISK LEVEL:
    Rate the overall policy as one of the following:
    - SAFE TO USE — minimal privacy risks found
    - SUSPICIOUS — some concerning clauses found, use with caution
    - HIGH RISK — serious privacy violations found, not recommended

    PRIVACY RISKS FOUND:
    List all potential privacy risks identified in the clauses.

    SUSPICIOUS CLAUSES:
    Highlight specific clauses that are concerning and explain why.

    RECOMMENDATIONS:
    Give the user practical advice based on the analysis.
    """
    return prompt

# Analyze privacy risk
def analyze_privacy_risk(question, text):
    try:
        chunks = retrieve_relevant_chunks(question, text)
        prompt = build_prompt(question, chunks)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize uploaded text in session state
if "policy_text" not in st.session_state:
    st.session_state.policy_text = ""

st.subheader("Upload Your Policy")

# Two tabs for upload options
tab1, tab2 = st.tabs(["📁 Upload TXT File", "✏️ Paste Text"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload your Terms & Conditions (.txt file)",
        type=["txt"]
    )
    if uploaded_file is not None:
        st.session_state.policy_text = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

with tab2:
    pasted_text = st.text_area(
        "Paste your Terms & Conditions here",
        height=200,
        placeholder="Paste your policy text here..."
    )
    if pasted_text:
        st.session_state.policy_text = pasted_text
        st.success("Text loaded successfully!")
st.subheader("Ask About Privacy Risks")

question = st.text_input(
    "Enter your question",
    placeholder="e.g. Does this policy share my data with third parties?"
)

analyze_button = st.button("Analyze Risk", use_container_width=True)

if analyze_button:
    if not st.session_state.policy_text:
        st.warning("Please upload or paste your Terms & Conditions first!")
    elif not question:
        st.warning("Please enter a question!")
    else:
        with st.spinner("Analyzing privacy risks..."):
            result = analyze_privacy_risk(question, st.session_state.policy_text)

            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result
            })
# Display latest result
if st.session_state.chat_history:
    latest = st.session_state.chat_history[-1]

    st.subheader("Analysis Result")

    # Risk level badge
    result_text = latest["answer"]
    if "HIGH RISK" in result_text:
        st.error("🔴 HIGH RISK — Serious privacy violations found!")
    elif "SUSPICIOUS" in result_text:
        st.warning("⚠️ SUSPICIOUS — Some concerning clauses found!")
    elif "SAFE TO USE" in result_text:
        st.success("✅ SAFE TO USE — Minimal privacy risks found!")

    # Full result
    st.markdown(latest["answer"])

    st.divider()
# Display chat history
if len(st.session_state.chat_history) > 1:
    st.subheader("💬 Previous Questions")

    # Show history in reverse order (latest first)
    for i, chat in enumerate(reversed(st.session_state.chat_history[:-1])):
        with st.expander(f"Q{len(st.session_state.chat_history) - 1 - i}: {chat['question']}"):
            # Risk badge for each history item
            if "HIGH RISK" in chat["answer"]:
                st.error("🔴 HIGH RISK")
            elif "SUSPICIOUS" in chat["answer"]:
                st.warning("⚠️ SUSPICIOUS")
            elif "SAFE TO USE" in chat["answer"]:
                st.success("✅ SAFE TO USE")

            st.markdown(chat["answer"])