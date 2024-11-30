import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

# Streamlitアプリのタイトル
st.title("PDF-based Chatbot")
st.write("Ask questions about a specific document.")

# モデルのロード
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# PDF知識ベースのロード
@st.cache_resource
def load_knowledge_base():
    with open("knowledge_base.pkl", "rb") as f:
        sentences, embeddings = pickle.load(f)
    return sentences, embeddings

# 質問に基づいて回答を取得
def get_answer(question, model, sentences, embeddings):
    question_embedding = model.encode(question)
    scores = util.pytorch_cos_sim(question_embedding, embeddings)
    best_match_idx = scores.argmax()
    return sentences[best_match_idx]

# モデルと知識ベースをロード
model = load_model()
sentences, embeddings = load_knowledge_base()

# ユーザー入力
question = st.text_input("Enter your question:")

# 回答生成
if question:
    try:
        answer = get_answer(question, model, sentences, embeddings)
        st.write(f"**Answer:** {answer}")
    except Exception as e:
        st.error("An error occurred while processing your question.")
        st.error(str(e))
修正箇所の詳細
