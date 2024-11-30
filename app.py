{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55805b35-7ec0-4a51-bc56-da0a0fa1ff9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pdfplumber\n",
    "from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, pipeline\n",
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
    "# DistilBERTを使った質問応答モデルの明示的なロード\n",
    "model_name = \"distilbert-base-cased-distilled-squad\"  # 使用するモデルを指定\n",
    "tokenizer = DistilBertTokenizer.from_pretrained(model_name)  # トークナイザーを読み込む\n",
    "model = DistilBertForQuestionAnswering.from_pretrained(model_name)  # モデルを読み込む\n",
    "\n",
    "# 質問応答のパイプラインを作成\n",
    "qa_pipeline = pipeline(\"question-answering\", model=model, tokenizer=tokenizer)\n",
    "\n",
    "# Streamlitアプリケーションの設定\n",
    "st.title(\"PDF質問応答システム\")\n",
    "st.write(\"このシステムは「Published Version」というPDF文書に基づいて質問に回答します。\")\n",
    "\n",
    "# PDFファイルのパスを指定（ローカルに保存されているPDFを使用）\n",
    "pdf_path = \"Published Version.pdf\"  # PDFファイルのパスを指定\n",
    "\n",
    "# PDFからテキストを抽出\n",
    "pdf_text = extract_text_from_pdf(pdf_path)\n",
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
