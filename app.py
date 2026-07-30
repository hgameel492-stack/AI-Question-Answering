import streamlit as st
from transformers import pipeline
import time

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="AI Question Answering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================
# Custom CSS
# =====================================
st.markdown(
    """
<style>

body{
    background:#f8fafc;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{
    background:linear-gradient(135deg,#2563eb,#4f46e5);
    padding:35px;
    border-radius:20px;
    color:white;
    text-align:center;
    margin-bottom:30px;
    box-shadow:0 8px 20px rgba(0,0,0,.15);
}

.hero h1{
    margin:0;
    font-size:42px;
}

.hero p{
    font-size:18px;
    color:#e5e7eb;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 5px 15px rgba(0,0,0,.08);
    margin-bottom:20px;
}

footer{
    visibility:hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# =====================================
# Load QA Model
# =====================================
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")


with st.spinner("🔄 Loading AI Model..."):
    qa_pipeline = load_model()

# =====================================
# Sidebar
# =====================================
with st.sidebar:

    st.title("🤖 AI Question Answering")

    st.markdown("---")

    st.subheader("📖 About")

    st.write("""
This application answers questions from any paragraph using
the **RoBERTa SQuAD2** model powered by Hugging Face.
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

    st.info("deepset/roberta-base-squad2")

    st.markdown("---")

    st.subheader("👩‍💻 Developer")

    st.success("Habiba Gamal")

    st.markdown("---")

    st.caption("Version 1.0")

# =====================================
# Hero Section
# =====================================
st.markdown(
    """
<div class="hero">

<h1>🤖 AI Question Answering</h1>

<p>
Ask questions about any paragraph using the powerful
<b>RoBERTa SQuAD2</b> model from Hugging Face.
</p>

</div>
""",
    unsafe_allow_html=True,
)
# =====================================
# Main Layout
# =====================================

left, right = st.columns([1.3, 1])

# =====================================
# Left Column (Input)
# =====================================

with left:

    st.subheader("📄 Context")

    context = st.text_area(
        label="", placeholder="Paste your paragraph here...", height=280
    )

    st.subheader("❓ Question")

    question = st.text_input(label="", placeholder="Ask your question...")

    # Statistics
    words = len(context.split())
    characters = len(context)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📝 Words", words)

    with col2:
        st.metric("🔠 Characters", characters)

    ask_button = st.button("🚀 Get Answer", use_container_width=True, type="primary")

# =====================================
# Right Column (Results)
# =====================================

with right:

    st.subheader("📊 Results")

    answer_placeholder = st.empty()

    confidence_placeholder = st.empty()

    response_placeholder = st.empty()

# =====================================
# Prediction
# =====================================

if ask_button:

    if context.strip() == "":

        st.warning("⚠ Please enter a context.")

    elif question.strip() == "":

        st.warning("⚠ Please enter a question.")

    else:

        with st.spinner("🤖 AI is thinking..."):

            start = time.time()

            result = qa_pipeline(question=question, context=context)

            end = time.time()

            answer = result["answer"]
            confidence = result["score"] * 100
            response_time = end - start

        with right:

            st.success("✅ Prediction Completed Successfully")

            st.markdown("### 💡 Answer")

            st.info(answer)

            st.markdown("### 📈 Confidence")

            st.progress(confidence / 100)

            st.metric("Confidence Score", f"{confidence:.2f}%")

            st.markdown("### ⚡ Response Time")

            st.metric("Execution Time", f"{response_time:.3f} sec")
            # =====================================
# Examples
# =====================================

st.markdown("---")
st.subheader("📚 Try an Example")

examples = {
    "Healthcare": (
        "Artificial intelligence is transforming healthcare by helping doctors analyze medical images and detect diseases earlier.",
        "How is AI helping healthcare?",
    ),
    "Egypt": (
        "Egypt is located in North Africa. Cairo is the capital of Egypt. The Nile River is the longest river in Africa.",
        "What is the capital of Egypt?",
    ),
    "Python": (
        "Python is a high-level programming language created by Guido van Rossum in 1991. It is widely used for web development, data science, and artificial intelligence.",
        "Who created Python?",
    ),
    "Eiffel Tower": (
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "Where is the Eiffel Tower located?",
    ),
    "Great Wall": (
        "The Great Wall of China was constructed to protect ancient Chinese states from invasions.",
        "Why was the Great Wall built?",
    ),
}

selected = st.selectbox("Choose an Example", list(examples.keys()))

col1, col2 = st.columns(2)

with col1:

    if st.button("📄 Show Example", use_container_width=True):

        example_context, example_question = examples[selected]

        st.info("### Context")
        st.write(example_context)

        st.info("### Question")
        st.write(example_question)

with col2:

    if st.button("🧹 Clear", use_container_width=True):
        st.rerun()

# =====================================
# Download Answer
# =====================================

if ask_button and context.strip() != "" and question.strip() != "":

    st.markdown("---")

    st.download_button(
        label="📥 Download Answer",
        data=f"""
Question:
{question}

Answer:
{answer}

Confidence:
{confidence:.2f}%

Response Time:
{response_time:.3f} sec
""",
        file_name="answer.txt",
        mime="text/plain",
        use_container_width=True,
    )

# =====================================
# Footer
# =====================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;padding:20px;'>

<h4>🤖 AI Question Answering</h4>

<p>
Powered by
<b>RoBERTa SQuAD2</b>
&
Streamlit
</p>

<p>
👩‍💻 Developed with ❤️ by
<b>Habiba Gamal</b>
</p>

</div>
""",
    unsafe_allow_html=True,
)
# =====================================
# Model Information
# =====================================

st.markdown("---")

with st.expander("🧠 About the Model"):

    st.markdown("""
### RoBERTa SQuAD2

This application uses the **deepset/roberta-base-squad2**
Question Answering model from Hugging Face.

#### Capabilities

- Answer questions from any paragraph
- Extractive Question Answering
- Fast inference
- High accuracy
- Based on RoBERTa architecture

#### Limitations

- The answer must exist inside the context.
- Very long contexts may increase inference time.
- Accuracy depends on context quality.
""")

# =====================================
# Tips
# =====================================

st.markdown("---")

with st.expander("💡 Tips for Better Results"):

    st.info("""
✔ Use a clear paragraph.

✔ Ask one question at a time.

✔ Make sure the answer exists in the paragraph.

✔ Avoid extremely long contexts.
""")

# =====================================
# Project Statistics
# =====================================

st.markdown("---")

st.subheader("📊 Project Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "RoBERTa")

with col2:
    st.metric("Task", "Question Answering")

with col3:
    st.metric("Framework", "Streamlit")

# =====================================
# Contact
# =====================================

st.markdown("---")

st.subheader("📬 Contact")

st.markdown("""
- 👩‍💻 **Developer:** Habiba Gamal
- 💻 **Framework:** Streamlit
- 🤗 **Model:** Hugging Face Transformers
""")

# =====================================
# Footer
# =====================================

st.markdown(
    """
---
<center>

Made with ❤️ using Streamlit & Hugging Face

© 2026 Habiba Gamal

</center>
""",
    unsafe_allow_html=True,
)
# =====================================
# Session Statistics
# =====================================

st.markdown("---")
st.subheader("📈 Session Statistics")

if "queries" not in st.session_state:
    st.session_state.queries = 0

if ask_button and context.strip() != "" and question.strip() != "":
    st.session_state.queries += 1

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Questions Answered", st.session_state.queries)

with col2:
    st.metric("Model", "RoBERTa")

with col3:
    st.metric("Status", "🟢 Online")

# =====================================
# AI Features
# =====================================

st.markdown("---")
st.subheader("🚀 Application Features")

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.success("""
✅ Fast Inference

Answer questions in seconds.
""")

with feature2:
    st.info("""
🧠 AI Powered

Powered by Hugging Face Transformers.
""")

with feature3:
    st.warning("""
🌍 Easy to Use

Simply paste your context and ask.
""")

# =====================================
# Developer Card
# =====================================

st.markdown("---")

st.markdown(
    """
<div style="background:#2563eb;
padding:25px;
border-radius:18px;
text-align:center;
color:white;">

<h2>👩‍💻 Habiba Gamal</h2>

<p>AI & Machine Learning Developer</p>

<p>
Python • Machine Learning • NLP • Streamlit
</p>

</div>
""",
    unsafe_allow_html=True,
)

# =====================================
# Feedback
# =====================================

st.markdown("---")

st.subheader("⭐ Rate this Application")

rating = st.slider("Your Rating", 1, 5, 5)

if st.button("Submit Rating"):
    st.success("🎉 Thank you for your feedback!")

# =====================================
# Footer
# =====================================

st.markdown("---")

st.caption("""
Made with ❤️ using Streamlit,
Hugging Face Transformers and PyTorch.

© 2026 Habiba Gamal
""")
