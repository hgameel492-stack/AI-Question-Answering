import streamlit as st
from transformers import pipeline
import time

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Question Answering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

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
margin-bottom:30px;
box-shadow:0 8px 20px rgba(0,0,0,.15);
}

.hero h1{
font-size:45px;
margin-bottom:10px;
}

.hero p{
font-size:18px;
color:#f3f4f6;
}

.metric-card{
background:white;
padding:15px;
border-radius:15px;
box-shadow:0 4px 12px rgba(0,0,0,.08);
text-align:center;
}

.answer-card{
background:#eff6ff;
padding:20px;
border-left:6px solid #2563eb;
border-radius:15px;
margin-top:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():

    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

with st.spinner("🤖 Loading AI Model..."):

    qa_pipeline = load_model()

if "context" not in st.session_state:
    st.session_state.context = ""

if "question" not in st.session_state:
    st.session_state.question = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "response_time" not in st.session_state:
    st.session_state.response_time = 0.0

# عدد الأسئلة
if "queries" not in st.session_state:
    st.session_state.queries = 0
    
        # ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.image("logo.png", width=120)

    st.title("🤖 AI Question Answering")

    st.markdown("---")

    st.subheader("📖 About")

    st.write(
        """
Ask questions about any paragraph using the
RoBERTa SQuAD2 model from Hugging Face.
"""
    )

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
        st.session_state.queries
    )
    # ==========================================
# Hero Section
# ==========================================

st.markdown("""
<div class="hero">

<h1>🤖 AI Question Answering</h1>

<p>
Extract answers from any paragraph using
<b>RoBERTa SQuAD2</b> powered by Hugging Face.
</p>

</div>
""", unsafe_allow_html=True)

# ==========================================
# Dashboard
# ==========================================

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

# ==========================================
# Main Layout
# ==========================================

left, right = st.columns([1.4, 1])

# ==========================================
# Left Column
# ==========================================

with left:

    st.subheader("📄 Context")

    st.session_state.context = st.text_area(
        "",
        value=st.session_state.context,
        height=300,
        placeholder="Paste your paragraph here..."
    )

    st.subheader("❓ Question")

    st.session_state.question = st.text_input(
        "",
        value=st.session_state.question,
        placeholder="Ask your question..."
    )

    words = len(st.session_state.context.split())
    chars = len(st.session_state.context)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("📝 Words", words)

    with c2:
        st.metric("🔠 Characters", chars)

    ask_button = st.button(
        "🚀 Get Answer",
        use_container_width=True,
        type="primary"
    )

# ==========================================
# Right Column
# ==========================================

with right:

    st.subheader("📊 Results")

    if st.session_state.answer:

        st.success("✅ Prediction Completed")

        st.markdown(f"""
<div class="answer-card">

<h3>💡 Answer</h3>

<p style="font-size:20px;">
{st.session_state.answer}
</p>

</div>
""", unsafe_allow_html=True)

        st.markdown("### 📈 Confidence")

        st.progress(st.session_state.confidence / 100)

        st.metric(
            "Confidence Score",
            f"{st.session_state.confidence:.2f}%"
        )

        st.metric(
            "⚡ Response Time",
            f"{st.session_state.response_time:.3f} sec"
        )

        st.code(st.session_state.answer)
# ==========================================
# Prediction
# ==========================================

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
        context=st.session_state.context
    )

    end = time.time()

    # حفظ النتائج
    st.session_state.answer = result["answer"]
    st.session_state.confidence = result["score"] * 100
    st.session_state.response_time = end - start

    # زيادة عدد الأسئلة
    st.session_state.queries += 1

# إعادة تشغيل الصفحة لعرض النتائج
st.rerun()


# ==========================================
# Download Report
# ==========================================

if st.session_state.answer:

    report = f"""
AI Question Answering Report

Question:
{st.session_state.question}

-------------------------------------

Answer:
{st.session_state.answer}

-------------------------------------

Confidence:
{st.session_state.confidence:.2f} %

-------------------------------------

Execution Time:
{st.session_state.response_time:.3f} sec
"""

    st.download_button(
        "📥 Download Report",
        report,
        "QA_Report.txt",
        "text/plain",
        use_container_width=True
    )
    
