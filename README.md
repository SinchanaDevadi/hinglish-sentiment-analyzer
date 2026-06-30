# 🌐 Hinglish Sentiment Analyzer

> Real-time sentiment classification for code-mixed Hindi-English (Hinglish) social media text using multilingual BERT — with gradient-based explainability.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

🔗 **[Live Demo →](https://your-app-name.streamlit.app)** ← *(replace after deploying)*

---

## 🧠 Why This Project?

Standard English sentiment tools fail completely on Hinglish — a real, widespread communication style used by 500M+ people across Indian social media. Existing NLP models either ignore the Hindi components or misclassify them entirely.

This project fine-tunes `bert-base-multilingual-cased` specifically on Hinglish text, making it one of the few open-source tools addressing this gap.

**Example inputs the model handles:**
- *"yaar ye movie ekdum bakwaas thi lol"* → Negative 😠
- *"bhai kal ka concert toh mast tha!"* → Positive 😄
- *"thoda theek tha, kuch khaas nahi"* → Neutral 😐

---

## ✨ Features

- ✅ 3-class sentiment prediction: **Positive / Negative / Neutral**
- ✅ Confidence score with visual bar chart per prediction
- ✅ **Token-level explainability** — highlights which words drove the prediction (gradient saliency)
- ✅ Interactive Streamlit UI — paste any Hinglish text and get instant results
- ✅ Handles Roman script Hindi, English, and mixed text

---

## 🏗 Project Structure

```
hinglish-sentiment-analyzer/
├── app.py               # Streamlit UI
├── train.py             # Fine-tuning script (run on GPU/Colab)
├── model_utils.py       # Tokenizer, inference, explainability
├── requirements.txt     # Dependencies
├── data/
│   └── README.md        # Dataset download instructions
└── saved_model/         # Fine-tuned model weights (after training)
```

---

## 📦 Dataset

**SemEval-2020 Task 9 — Hinglish Sentiment Analysis**
- Download from: https://ritual.uh.edu/semeval2020-task9/
- Place files as `data/Hinglish_train.csv` and `data/Hinglish_dev.csv`
- Columns: `text`, `label` (Positive / Negative / Neutral)

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/SinchanaDevadi/hinglish-sentiment-analyzer.git
cd hinglish-sentiment-analyzer
pip install -r requirements.txt
```

### 2. Train the model (Google Colab recommended — free GPU)
```bash
# Download dataset first (see data/README.md)
python train.py
# Model saves to saved_model/ automatically
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🧪 Model Details

| Component | Detail |
|-----------|--------|
| Base Model | `bert-base-multilingual-cased` |
| Framework | PyTorch + HuggingFace Transformers |
| Task | 3-class sequence classification |
| Explainability | Gradient-based saliency (token importance) |
| Training | 4 epochs, lr=2e-5, batch size=16 |
| Deployment | Streamlit Cloud |

---

## 📸 Screenshots

*(Add screenshots of your Streamlit app here after running it)*

---

## 🙋 Author

**Sinchana H Devadiga**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/sinchana-devadiga)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:sinchanadevadiga360@gmail.com)
