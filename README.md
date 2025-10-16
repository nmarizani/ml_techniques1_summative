# Mental Health Chatbot — FLAN-T5 (Healthcare Domain)

A supportive, domain-specific chatbot designed to provide empathetic and informative responses to mental-health–related questions.  
Built using Google FLAN-T5, fine-tuned on a custom mental-health conversational dataset, with the model hosted on Google Drive.


## Project Overview

This project demonstrates the implementation of a domain-specific Transformer chatbot focused on mental health awareness and self-care.  
The chatbot generates human-like responses to user queries about mental wellness using a fine-tuned generative language model (FLAN-T5).

The system provides a simple, intuitive web interface via Streamlit, enabling real-time user interaction with the AI model.


## Objectives

- Fine-tune a Transformer-based language model (FLAN-T5) on mental-health–related conversations.  
- Build a responsive streamlit app that interacts with the model in real time.  
- Ensure empathetic and contextually accurate responses to user inputs.  
- Deploy a fully functional, user-friendly chatbot for demonstration.


## Project Structure
app1.py # Streamlit application

dataset.csv # Domain-specific dataset (uploaded)

requirements.txt # Dependencies for Streamlit Cloud

best_flan_t5_healthcare_model/ # Fine-tuned model (linked from Drive)

README.md # Project documentation


## Dataset

The dataset (`dataset.csv`) contains conversational pairs between users and a mental-health assistant:

| Column   | Description |
|-----------|-------------|
| Context   | User question or statement |
| Response  | Appropriate chatbot reply |

**Preprocessing applied:**
- Cleaning (removed duplicates, nulls, irrelevant text)
- Normalization (lowercasing, punctuation removal)
- Tokenization via `AutoTokenizer` (Hugging Face)
- Padding and truncation for consistent input length


##  Model Details

- **Base Model:** `google/flan-t5-base`
- **Framework:** TensorFlow + Hugging Face Transformers
- **Fine-Tuning Parameters:**
  - Learning Rate: `3e-5`
  - Batch Size: `8`
  - Epochs: `3`
  - Optimizer: `AdamW`
- **Generation Settings:**
  - `max_new_tokens=120`
  - `num_beams=4`
  - `repetition_penalty=2.0`
  - `temperature=0.7`
  - `early_stopping=True`

The model generates contextually coherent and emotionally sensitive responses suitable for mental-health conversations.


## Streamlit App Features

- **Chat Interface:** Real-time conversation with the chatbot  
- **Clear Chat:** Option to reset the session  
- **Professional Help Buttons:** Simulated “Call” and “Message” options  
- **Google Drive Model Integration:** Automatically loads model weights from Drive (for large file compatibility)  
- **Responsive UI:** Modern and accessible chat layout for all users  


## How It Works

1. **User Input:** The user types a message related to mental health.  
2. **Context Matching:** If similar entries exist in the dataset, the chatbot uses them for a fast, relevant reply.  
3. **Generative Response:** Otherwise, the model generates a free-text answer using the fine-tuned FLAN-T5.  
4. **Display:** The chatbot’s response appears in a visually appealing chat bubble.  



### Run On Virtual Environment

Create a virtual environment and activate it then run the following commands

/.venv/Scripts/activate

pip install -r requirements.txt

streamlit run app.py

### Project Links

GitHub Repo: https://github.com/nmarizani/ml_techniques1_summative.git

Video Demo: https://drive.google.com/file/d/13uiQN7jrHQAXgf1TABGBz4quIIYjPR-7/view?usp=sharing

