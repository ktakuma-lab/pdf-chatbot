{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "caf020fe-b43f-4f92-b4d5-b011e3d12008",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No model was supplied, defaulted to distilbert/distilbert-base-cased-distilled-squad and revision 564e9b5 (https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad).\n",
      "Using a pipeline without specifying a model name and revision in production is not recommended.\n"
     ]
    }
   ],
   "source": [
    "import pdfplumber\n",
    "from transformers import pipeline\n",
    "import streamlit as st\n",
    "\n",
    "# PDFからテキストを抽出する関数\n",
    "def extract_text_from_pdf(pdf_path):\n",
    "    with pdfplumber.open(pdf_path) as pdf:\n",
    "        text = \"\"\n",
    "        for page in pdf.pages:\n",
    "            text += page.extract_text()\n",
    "    return text\n",
    "\n",
    "# 質問応答のパイプラインの作成\n",
    "qa_pipeline = pipeline(\"question-answering\")\n",
    "\n",
    "# PDFファイルのパスを指定（ローカルに保存されているPDFを使用）\n",
    "pdf_path = \"Published Version.pdf\"  # PDFファイルのパスを指定\n",
    "\n",
    "# PDFからテキストを抽出\n",
    "pdf_text = extract_text_from_pdf(pdf_path)\n",
    "\n",
    "# Streamlitアプリケーションの設定\n",
    "st.title(\"PDF質問応答システム\")\n",
    "st.write(\"このシステムは「Published Version」というPDF文書に基づいて質問に回答します。\")\n",
    "\n",
    "# ユーザーからの質問を受け取る\n",
    "question = st.text_input(\"質問を入力してください:\")\n",
    "\n",
    "if question:\n",
    "    # 質問に対する回答を生成\n",
    "    result = qa_pipeline(question=question, context=pdf_text)\n",
    "    answer = result['answer']\n",
    "    \n",
    "    # 回答を表示\n",
    "    st.write(f\"回答: {answer}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0812da1a-94a5-473a-a099-6e1d7dde9950",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
