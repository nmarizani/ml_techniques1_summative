import os
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="wide")

# Load fine-tuned model
MODEL_DIR = "best_flan_t5_healthcare_model"
if not os.path.exists(MODEL_DIR):
    st.error("Model folder not found. Upload or mount it first.")
    st.stop()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

DATASET_PATH = "dataset.csv"
if os.path.exists(DATASET_PATH):
    df_faq = pd.read_csv(DATASET_PATH)
    df_faq.columns = df_faq.columns.str.strip()
else:
    df_faq = pd.DataFrame(columns=["Context", "Response"])

# Generation settings
GEN_ARGS = dict(
    max_new_tokens=120,
    num_beams=4,
    repetition_penalty=2.0,
    no_repeat_ngram_size=3,
    temperature=0.7,
    early_stopping=True,
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Response function
def generate_response(user_input):
    if not user_input.strip():
        return "Please type something so I can help you."

    if not df_faq.empty:
        matches = df_faq[df_faq["Context"].str.contains(user_input, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0]["Response"]

    # Use FLAN-T5
    inputs = tokenizer(user_input, return_tensors="tf", truncation=True, padding=True, max_length=256)
    outputs = model.generate(**inputs, **GEN_ARGS)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# UI
st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
}
.user-msg {
    background-color: #6c63ff;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    text-align: left;
    margin: 5px 0px 5px 40px;
    display: inline-block;
}
.bot-msg {
    background-color: #f1f0f0;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    text-align: left;
    margin: 5px 40px 5px 0px;
    display: inline-block;
}
.timestamp {
    font-size: 0.7em;
    color: gray;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Health Chatbot")
st.markdown("A supportive chatbot powered by your fine-tuned FLAN-T5 model.")

# Input
user_input = st.text_input("Type your message here:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Send"):
        if user_input:
            response = generate_response(user_input)
            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Wellness Friend", response))
with col2:
    if st.button("📞 Speak to a Professional"):
        st.session_state.history.append(("Wellness Friend", "Connecting you to a mental health professional..."))
with col3:
    if st.button("📱 Call Someone"):
        st.session_state.history.append(("Wellness Friend", "If you need urgent help, please call your local helpline: 123-456-7890"))

# Clear chat
if st.button("Clear Chat"):
    st.session_state.history = []

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for speaker, message in st.session_state.history:
    timestamp = datetime.now().strftime("%H:%M")
    if speaker == "You":
        st.markdown(f'<div class="user-msg">{message}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{message}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
