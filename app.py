#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdfplumber
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, pipeline
import streamlit as st

# PDFからテキストを抽出する関数
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# DistilBERTを使った質問応答モデルの明示的なロード
model_name = "distilbert-base-cased-distilled-squad"  # 使用するモデルを指定
tokenizer = DistilBertTokenizer.from_pretrained(model_name)  # トークナイザーを読み込む
model = DistilBertForQuestionAnswering.from_pretrained(model_name)  # モデルを読み込む

# 質問応答のパイプラインを作成
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Streamlitアプリケーションの設定
st.title("PDF質問応答システム")
st.write("このシステムは「Published Version」というPDF文書に基づいて質問に回答します。")

# PDFファイルのパスを指定（ローカルに保存されているPDFを使用）
pdf_path = "Published Version.pdf"  # PDFファイルのパスを指定

# PDFからテキストを抽出
pdf_text = extract_text_from_pdf(pdf_path)

# ユーザーからの質問を受け取る
question = st.text_input("質問を入力してください:")

if question:
    # 質問に対する回答を生成
    result = qa_pipeline(question=question, context=pdf_text)
    answer = result['answer']
    
    # 回答を表示
    st.write(f"回答: {answer}")


# In[ ]:




