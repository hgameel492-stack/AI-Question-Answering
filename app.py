import streamlit as st
from transformers import pipeline
import time

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AI Question Answering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# Custom CSS
# =====================================================

st.markdown(
    """
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{
    background:linear-gradient(135deg,#2563eb,#4f46e5);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:25px;
    box-shadow:0 10px 25px rgba(0,0,0,.15);
}

.hero h1{
    font-size:46px;
    margin-bottom:8px;
}

.hero p{
    font-size:18px;
    color:#f8fafc;
}

.answer-card{
    background:#eff6ff;
    padding:20px;
    border-left:6px solid #2563eb;
    border-radius:15px;
    margin-top:15px;
}

.example-box{
    background:#f8fafc;
    border-radius:12px;
    padding:15px;
    margin-top:10px;
}

</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# Load Model
# =====================================================


@st.cache_resource
def load_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
    )


with st.spinner("🤖 Loading AI Model..."):
    qa_pipeline = load_model()

# =====================================================
# Session State
# =====================================================

defaults = {
    "context": "",
    "question": "",
    "answer": "",
    "confidence": 0.0,
    "response_time": 0.0,
    "queries": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.image("logo.jpg", width=120)

    st.title("🤖 AI Question Answering")

    st.markdown("---")

    st.subheader("📖 About")

    st.write("""
Ask questions about any paragraph using
RoBERTa SQuAD2 from Hugging Face.
""")

    st.markdown("---")

    st.subheader("⚙ Technologies")

    st.markdown("""
- Python
- Streamlit
- Hugging Face
- Transformers
- PyTorch
""")

    st.markdown("---")

    st.subheader("🧠 Model")
    st.success("deepset/roberta-base-squad2")

    st.markdown("---")

    st.subheader("👩‍💻 Developer")
    st.info("Habiba Gamal")

    st.markdown("---")

    st.metric(
        "Questions Answered",
        st.session_state.queries,
    )

# =====================================================
# Hero Section
# =====================================================

st.markdown(
    """
<div class="hero">

<h1>🤖 AI Question Answering</h1>

<p>
Extract answers from any paragraph using
<b>RoBERTa SQuAD2</b> powered by Hugging Face.
</p>

</div>
""",
    unsafe_allow_html=True,
)

# =====================================================
# Dashboard
# =====================================================

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric("🧠 Model", "RoBERTa")

with d2:
    st.metric("📄 Task", "Question Answering")

with d3:
    st.metric("⚡ Status", "Online")

with d4:
    st.metric("📊 Questions", st.session_state.queries)

st.markdown("---")
# =====================================================
# Main Layout
# =====================================================

left, right = st.columns([1.4, 1])

# =====================================================
# Left Column
# =====================================================

with left:

    st.subheader("📄 Context")

    st.session_state.context = st.text_area(
        "",
        value=st.session_state.context,
        height=280,
        placeholder="Paste your paragraph here...",
    )

    st.subheader("❓ Question")

    st.session_state.question = st.text_input(
        "",
        value=st.session_state.question,
        placeholder="Ask your question...",
    )

    # -------------------------
    # Statistics
    # -------------------------

    words = len(st.session_state.context.split())
    chars = len(st.session_state.context)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("📝 Words", words)

    with c2:
        st.metric("🔠 Characters", chars)

    st.markdown("---")

    # =====================================================
    # Examples
    # =====================================================

    st.subheader("📝 Quick Examples")

    e1, e2 = st.columns(2)

    with e1:

        if st.button("🌍 Geography", use_container_width=True):

            st.session_state.context = (
                "Paris is the capital city of France. "
                "It is famous for the Eiffel Tower."
            )

            st.session_state.question = "What is the capital of France?"

            st.rerun()

        if st.button("🤖 Artificial Intelligence", use_container_width=True):

            st.session_state.context = (
                "Artificial Intelligence is a branch of computer science "
                "that enables machines to perform tasks requiring human intelligence."
            )

            st.session_state.question = "What is Artificial Intelligence?"

            st.rerun()

    with e2:

        if st.button("🐍 Python", use_container_width=True):

            st.session_state.context = (
                "Python is a popular programming language used in "
                "Machine Learning, Data Science, Web Development, and AI."
            )

            st.session_state.question = "What is Python used for?"

            st.rerun()

        if st.button("📜 History", use_container_width=True):

            st.session_state.context = (
                "The Great Wall of China was built to protect Chinese states "
                "against invasions and raids."
            )

            st.session_state.question = "Why was the Great Wall of China built?"

            st.rerun()

    st.markdown("---")

    # =====================================================
    # Buttons
    # =====================================================

    b1, b2 = st.columns(2)

    with b1:

        ask_button = st.button(
            "🚀 Get Answer",
            use_container_width=True,
            type="primary",
        )

    with b2:

        clear_button = st.button(
            "🗑 Clear",
            use_container_width=True,
        )

    # =====================================================
    # Clear
    # =====================================================

    if clear_button:

        st.session_state.context = ""
        st.session_state.question = ""
        st.session_state.answer = ""
        st.session_state.confidence = 0.0
        st.session_state.response_time = 0.0

        st.rerun()

# =====================================================
# Right Column
# =====================================================

with right:

    st.subheader("📊 Results")

    if st.session_state.answer:

        st.success("✅ Prediction Completed")

        st.markdown(
            f"""
<div class="answer-card">

<h3>💡 Answer</h3>

<p style="font-size:20px;">
{st.session_state.answer}
</p>

</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### 📈 Confidence")

        st.progress(st.session_state.confidence / 100)

        st.metric("Confidence", f"{st.session_state.confidence:.2f}%")

        st.metric("⚡ Response Time", f"{st.session_state.response_time:.3f} sec")

        st.code(
            st.session_state.answer,
            language=None,
        )
# =====================================================
# Prediction
# =====================================================

if ask_button:

    if st.session_state.context.strip() == "":
        st.warning("⚠ Please enter a context.")

    elif st.session_state.question.strip() == "":
        st.warning("⚠ Please enter a question.")

    else:

        with st.spinner("🤖 AI is thinking..."):

            start = time.time()

            result = qa_pipeline(
                question=st.session_state.question,
                context=st.session_state.context,
            )

            end = time.time()

            answer = result["answer"].strip()

            if answer == "":
                st.session_state.answer = "No answer found."
            else:
                st.session_state.answer = answer

            st.session_state.confidence = result["score"] * 100
            st.session_state.response_time = end - start
            st.session_state.queries += 1

            st.rerun()

# =====================================================
# Download Report
# =====================================================

if st.session_state.answer:

    st.markdown("---")

    report = f"""
=============================
AI Question Answering Report
=============================

Question
---------
{st.session_state.question}

Answer
------
{st.session_state.answer}

Confidence
----------
{st.session_state.confidence:.2f} %

Response Time
-------------
{st.session_state.response_time:.3f} sec

Model
-----
deepset/roberta-base-squad2
"""

    c1, c2 = st.columns(2)

    with c1:

        st.download_button(
            "📥 Download Report",
            data=report,
            file_name="QA_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with c2:

        st.download_button(
            "📋 Download Answer",
            data=st.session_state.answer,
            file_name="answer.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.markdown("---")

st.caption("🚀 Built with Streamlit + Hugging Face Transformers")
