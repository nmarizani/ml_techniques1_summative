import os
import torch
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# LOAD FINE-TUNED MODEL
MODEL_DIR = "best_flan_t5_healthcare_model"
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError("Model folder 'best_flan_t5_healthcare_model' not found. Upload or mount it first.")

print(f"Loading fine-tuned model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# LOAD DATASET FOR QUICK RESPONSES
DATASET_PATH = "dataset.csv"
if os.path.exists(DATASET_PATH):
    df_faq = pd.read_csv(DATASET_PATH)
    df_faq.columns = df_faq.columns.str.strip()  # Clean column names
    print(f"Loaded dataset with {len(df_faq)} entries.")
else:
    print("Dataset not found. Creating empty DataFrame.")
    df_faq = pd.DataFrame(columns=["Context", "Response"])

# GENERATION SETTINGS
GEN_ARGS = dict(
    max_new_tokens=120,
    num_beams=4,
    repetition_penalty=2.0,
    no_repeat_ngram_size=3,
    temperature=0.7,
    early_stopping=True,
)

conversation_history = []

# RESPONSE GENERATION FUNCTION
def generate_response(user_input):
    if not user_input.strip():
        return "Please type something so I can help you."

    # Check dataset first 
    if not df_faq.empty:
        # Case-insensitive search in 'Context' column
        matches = df_faq[df_faq["Context"].str.contains(user_input, case=False, na=False)]
        if not matches.empty:
            response = matches.iloc[0]["Response"]
            conversation_history.append({"User": user_input, "Bot": response})
            return response

    # If not found, use fine-tuned FLAN-T5 model ---
    inputs = tokenizer(user_input, return_tensors="tf", truncation=True, padding=True, max_length=256)
    outputs = model.generate(**inputs, **GEN_ARGS)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    conversation_history.append({"User": user_input, "Bot": response})
    return response

# BUILD GRADIO UI
with gr.Blocks(css="""
    body {background-color: #eef3f7;}
    .chatbot {border-radius: 12px; padding: 15px; background-color: #ffffff; 
              box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    .action-btn {background-color: #6c63ff; color: white; padding: 10px 16px; 
                 border-radius: 8px; font-weight: 600;}
    .action-btn:hover {background-color: #5a54e8;}
""") as app:

    # Header Section
    gr.Markdown(
        """
        <div style="text-align:center;">
            <h1 style="color:#2c3e50;">🧠 Mental Health Chatbot</h1>
            <p style="color:#34495e;">
            A supportive chatbot powered by your fine-tuned <b>FLAN-T5</b> model.<br>
            It uses learned knowledge for instant answers and connects you with help if needed.
            </p>
        </div>
        """,
        elem_id="header"
    )

    # Chat Section
    chatbox = gr.Chatbot(label="Chat with me", elem_classes="chatbot")

    # Input Row
    with gr.Row():
        user_input = gr.Textbox(label="Type your message", placeholder="e.g., How can I manage anxiety?")
        send_btn = gr.Button("Send", elem_classes="action-btn")
        clear_btn = gr.Button("Clear Chat", elem_classes="action-btn")

    # Action Buttons
    with gr.Row():
        speak_btn = gr.Button("📞 Speak to a Professional", elem_classes="action-btn")
        call_btn = gr.Button("📱 Call Someone", elem_classes="action-btn")

    # Chat Function
    def chat_fn(message, history):
        response = generate_response(message)
        history.append((message, response))
        return "", history

    # Button Behaviors
    user_input.submit(chat_fn, [user_input, chatbox], [user_input, chatbox])
    send_btn.click(chat_fn, [user_input, chatbox], [user_input, chatbox])
    clear_btn.click(lambda: [], None, chatbox)

    speak_btn.click(lambda: ("Connecting you to a professional...", []), None, chatbox)
    call_btn.click(lambda: ("If you need urgent help, please call your local helpline: 123-456-7890", []), None, chatbox)

app.launch(share=True)