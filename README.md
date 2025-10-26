# 🗣️ Urdu Conversational Chatbot  
### Transformer-Based Model with Multi-Head Attention (Built from Scratch)

---

## 🌟 Overview

This repository contains the implementation of a **Transformer-based Urdu Conversational Chatbot**, trained **from scratch** without any pre-trained models.  
The model is capable of generating **context-aware Urdu responses** using **multi-head attention**, **positional encoding**, and a **custom encoder-decoder architecture**.  

A **Streamlit** (or **Gradio**) interface allows users to interact with the chatbot in real-time using Urdu input.

---

## 🚀 Features

- 🧠 Transformer Encoder–Decoder architecture implemented from scratch (PyTorch)
- 🔄 Multi-Head Attention mechanism for contextual understanding
- 🗂️ Custom Urdu dataset preprocessing and tokenization
- 📊 Training with teacher forcing and BLEU-based model checkpointing
- 💬 Real-time Urdu chat interface (Streamlit or Gradio)
- 🕋 Proper right-to-left (RTL) Urdu text rendering
- 📈 Evaluation using BLEU, ROUGE-L, chrF, and Perplexity

---

## 🧹 Data Preprocessing

1. **Normalization:**  
   Remove Urdu diacritics, unify Alef and Yeh forms.

2. **Tokenization:**  
   Custom Urdu tokenizer built with regex + frequency-based vocab generation.

3. **Vocabulary:**  
   - `PAD`, `SOS`, `EOS`, and `UNK` tokens added.  
   - Indexed vocabulary saved as `vocab.pkl`.

4. **Splits:**
   - Train → 80%  
   - Validation → 10%  
   - Test → 10%

---

## 🧠 Model Architecture

**Implemented Components:**
- Embedding Layer  
- Positional Encoding  
- Multi-Head Attention  
- Feed Forward Networks  
- Layer Normalization & Residual Connections  
- Encoder–Decoder with causal masking  

### 🔧 Suggested Hyperparameters

| Parameter | Value |
|------------|--------|
| Embedding Dim | 512 |
| Attention Heads | 2 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Dropout | 0.2 |
| Batch Size | 64 |
| Learning Rate | 3e-4 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

---

## 🏋️ Training

During training:

* Teacher forcing is used.
* BLEU score on validation set is monitored.
* The model with the **best BLEU** score is saved in `/checkpoints/best_model.pth`.

---

## 📈 Evaluation

**Metrics:**

* **BLEU**
* **ROUGE-L**
* **chrF**
* **Perplexity**

**Human Evaluation:**
Manual rating for **Fluency**, **Relevance**, and **Adequacy** on a 1–5 scale.

### 💻 Features

* Urdu text input box (right-to-left)
* Choice of decoding method:

  * Greedy decoding
  * Beam search
* Real-time response generation

---

## ☁️ Deployment

You can deploy your chatbot on:

* [Streamlit Cloud](https://streamlit.io/cloud)
  or
* [Gradio Spaces](https://huggingface.co/spaces)

Ensure **Urdu fonts** and **RTL support** are enabled in deployment.

---

## 🧪 Results Summary

| Metric     | Score |
| ---------- | ----- |
| BLEU       | 27.6  |
| ROUGE-L    | 0.48  |
| chrF       | 0.54  |
| Perplexity | 18.9  |

**Human Evaluation:**
Average fluency and relevance rated above 4.2/5.

---

## 📰 Blog Post

A detailed write-up covering:

* Dataset analysis
* Model architecture
* Training pipeline
* Evaluation and results
* Demo screenshots

📖 [Read the Medium Post](#) *(Add link here)*

---

## 🧑‍💻 Authors

| Name      | Roll No. |
| --------- | -------- |
| Student 1 | 22F-3876 |
| Student 2 | 22F-3350 |

---

## 🧩 Technologies Used

* **Python 3.10+**
* **PyTorch**
* **NumPy**, **Pandas**
* **NLTK / UrduHack**
* **Streamlit / Gradio**
* **scikit-learn**
* **sacrebleu**, **rouge-score**

---

## 🏁 Future Improvements

* Integrate response length control
* Visualize attention maps for Urdu text
* Experiment with deeper Transformer layers
* Add multilingual support

---
