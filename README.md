#  AI News Intelligence

An AI-powered Fake News Detection and News Intelligence application with a **responsive, mobile-friendly web interface**. It uses Natural Language Processing (NLP), Machine Learning, Deep Learning, and Big Data Analytics to analyze news articles and classify them as **Fake** or **Real**.

The application supports multiple prediction approaches and input formats, including **text, PDF, and image-based news analysis**.

---

##  Live Demo

###  Render Deployment

https://ai-news-intelligence-5nka.onrender.com/#analyzer

###  GitHub Repository

https://github.com/HiKomal11/AI-News-Intelligence

###  Local Development

```text
http://127.0.0.1:5000
```

---

##  Features

###  Fake News Detection

The application analyzes news content and predicts whether it is:

- 🟢 **Real**
- 🔴 **Fake**

The system currently uses two trained models:

1. **TF-IDF + Logistic Regression**
2. **LSTM Neural Network**

---

###  Multiple Input Formats

The application supports:

- **Text** — Enter or paste a news article directly.
- **PDF** — Upload a PDF containing a news article.
- **Image** — Upload a screenshot or image of a news article.

For PDF and image inputs, the application extracts the text before sending it through the NLP and prediction pipeline.

---

###  NLP Processing

The project includes a dedicated NLP preprocessing pipeline for preparing news text before model prediction and training.

The preprocessing workflow includes operations such as:

- Text cleaning
- Lowercase conversion
- Removing unnecessary characters
- Tokenization
- Stopword handling
- Text normalization

---

###  Big Data Analytics

The project also includes Big Data analysis and visualization using **PySpark**.

The analysis includes:

- Label distribution
- Subject distribution
- Subject vs. label distribution
- Dataset summary
- News category visualization
- Dataset statistics

Big Data analysis outputs are stored inside:

```text
results/
```

---

###  Model Evaluation

The trained models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

###  Responsive & Mobile-Friendly Interface

The application provides a responsive user interface designed to work across:

-  Desktop computers
-  Mobile phones
-  Tablets

The layout automatically adapts to different screen sizes, making the news analyzer, model selection, file upload, prediction results, and other interface components accessible on smaller screens.

---

##  Machine Learning Model

### TF-IDF + Logistic Regression

The first classification approach uses **TF-IDF feature extraction** followed by **Logistic Regression**.

### Workflow

```text
News Article
     ↓
NLP Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Fake / Real
```

### Configuration

| Parameter | Value |
|---|---|
| Maximum Features | 50,000 |
| N-gram Range | (1, 2) |
| Minimum Document Frequency | 2 |
| Maximum Document Frequency | 0.95 |
| Sublinear TF | Enabled |
| Classifier | Logistic Regression |
| Maximum Iterations | 1000 |
| Class Weight | Balanced |

### Performance

The trained TF-IDF + Logistic Regression model achieved:

**Accuracy: 99.10%**

### Confusion Matrix

```text
                Predicted
              Fake    Real

Actual Fake   3526     55
Actual Real     15   4224
```

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Fake | 1.00 | 0.98 | 0.99 |
| Real | 0.99 | 1.00 | 0.99 |

---

##  Deep Learning Model

### LSTM News Classifier

The second classification approach uses a **Long Short-Term Memory (LSTM)** neural network for sequential text classification.

The project also includes a **TensorFlow Lite version** of the LSTM model for lightweight inference and deployment.

### Workflow

```text
News Article
     ↓
Tokenization
     ↓
Sequence Conversion
     ↓
Padding
     ↓
Embedding
     ↓
LSTM
     ↓
Dropout
     ↓
Dense Layer
     ↓
Fake / Real
```

### Configuration

| Parameter | Value |
|---|---|
| Maximum Vocabulary | 10,000 words |
| Maximum Sequence Length | 200 |
| Embedding Dimension | 64 |
| LSTM Units | 32 |
| Epochs | 8 |
| Batch Size | 64 |
| Dropout | 0.4 / 0.2 |
| Early Stopping | Enabled |

### Performance

The trained LSTM model achieved:

**Accuracy: 96.36%**

### Confusion Matrix

```text
                Predicted
              Fake    Real

Actual Fake   3379    202
Actual Real     83   4156
```

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Fake | 0.98 | 0.94 | 0.96 |
| Real | 0.95 | 0.98 | 0.97 |

---

##  Model Comparison

| Model | Accuracy | Approach |
|---|---:|---|
| TF-IDF + Logistic Regression | **99.10%** | Machine Learning |
| LSTM | **96.36%** | Deep Learning |

Based on the current held-out test split, **TF-IDF + Logistic Regression provides the higher classification accuracy**, while the LSTM provides a neural-network-based sequential text classification approach.

Both models were evaluated on the same held-out test set.

---

##  Lightweight LSTM Deployment

The project includes a lightweight deployment workflow using **TensorFlow Lite**.

The trained Keras LSTM model can be converted into a TensorFlow Lite model for lightweight inference and reduced deployment requirements.

### Convert the LSTM Model

```bash
python convert_lstm_tflite.py
```

The resulting TensorFlow Lite model is:

```text
models/lstm_news_classifier.tflite
```

### Test the TensorFlow Lite Model

```bash
python test_lstm_tflite.py
```

The project therefore contains:

- Full Keras LSTM model
- TensorFlow Lite LSTM model for lightweight inference

---

##  Big Data Analytics

The project uses **PySpark** for Big Data analysis.

The analysis generates information related to:

### Label Distribution

Distribution of Fake and Real news.

### Subject Distribution

Distribution of news articles across different subjects/categories.

### Subject vs. Label Distribution

Analysis of how Fake and Real labels are distributed across different news subjects.

### Output Files

The generated results include:

```text
results/
├── bigdata_label_distribution.csv
├── bigdata_label_distribution.png
├── bigdata_subject_distribution.csv
├── bigdata_subject_distribution.png
├── bigdata_subject_label_distribution.csv
└── bigdata_summary.txt
```

---

##  Project Structure

```text
AI-News-Intelligence/
│
├── data/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lstm_news_classifier.keras
│   ├── lstm_news_classifier.tflite
│   ├── lstm_news_classifier_backup.keras
│   ├── lstm_tokenizer.pkl
│   ├── lstm_tokenizer_backup.pkl
│   └── lstm_word_index.pkl
│
├── notebooks/
│
├── results/
│   ├── bigdata_label_distribution.csv
│   ├── bigdata_label_distribution.png
│   ├── bigdata_subject_distribution.csv
│   ├── bigdata_subject_distribution.png
│   ├── bigdata_subject_label_distribution.csv
│   ├── bigdata_summary.txt
│   └── generated/
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── nlp_preprocessing.py
│   │   ├── prepare_data.py
│   │   └── text_cleaner.py
│   │
│   ├── training/
│   │   ├── train_tfidf.py
│   │   └── train_lstm.py
│   │
│   ├── __init__.py
│   ├── bigdata_analysis.py
│   ├── bigdata_results.py
│   ├── bigdata_test.py
│   ├── eda.py
│   ├── inspect_data.py
│   ├── upload_utils.py
│   └── visualize_bigdata.py
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│   └── .gitkeep
│
├── app.py
├── Dockerfile
├── requirements.txt
├── convert_lstm_tflite.py
├── test_lstm_tflite.py
│
├── article_length_distribution.png
├── class_distribution.png
├── tfidf_confusion_matrix.png
├── lstm_accuracy.png
└── lstm_confusion_matrix.png
```

---

##  Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- TF-IDF
- Logistic Regression

### Deep Learning

- TensorFlow
- Keras
- LSTM
- TensorFlow Lite

### Natural Language Processing

- NLTK
- Text preprocessing
- Tokenization

### Data Processing

- Pandas
- NumPy

### Big Data

- PySpark

### Visualization

- Matplotlib

### Web Application

- Flask
- HTML
- CSS
- JavaScript
- Responsive Web Design
- Mobile-friendly UI

### Deployment

- Render
- Docker

### Version Control

- Git
- GitHub

---

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HiKomal11/AI-News-Intelligence.git
```

### 2. Move into the Project Directory

```bash
cd AI-News-Intelligence
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment on Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Run Locally

Start the Flask application:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

---

##  Model Training

### Train the TF-IDF Model

```bash
python src/training/train_tfidf.py
```

This generates:

```text
models/tfidf_vectorizer.pkl
models/logistic_regression.pkl
```

It also generates the TF-IDF confusion matrix:

```text
tfidf_confusion_matrix.png
```

### Train the LSTM Model

```bash
python src/training/train_lstm.py
```

This generates:

```text
models/lstm_news_classifier.keras
models/lstm_tokenizer.pkl
```

The training process also generates:

```text
lstm_accuracy.png
lstm_confusion_matrix.png
```

---

##  Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The current evaluation results are:

| Model | Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | **99.10%** |
| LSTM | **96.36%** |

These results are based on the current training configuration and held-out test split.

---

##  Objective

The main objective of this project is to develop an intelligent news analysis system capable of automatically classifying potentially fake or misleading news using **Natural Language Processing, Machine Learning, and Deep Learning** techniques.

The project combines:

- Machine Learning
- Deep Learning
- NLP
- Big Data Analytics
- Multi-format news analysis
- TensorFlow Lite
- Flask web deployment
- Responsive web design

into a single application.

---

##  Future Enhancements

Possible future improvements include:

- Transformer-based models such as BERT
- Multilingual fake news detection
- Real-time news verification
- News source credibility analysis
- Explainable AI
- Improved OCR for image-based news
- Browser extension for instant news verification
- Advanced ensemble models
- Further LSTM optimization
- Improved lightweight deployment

---

##  Disclaimer

This application provides an **AI-based classification of news content** and should not be considered a definitive fact-checking or news-verification system.

A prediction of **Real** or **Fake** represents the model's classification based on patterns learned from its training data. Users should verify important information through reliable and independent sources.

---

##  Author

**Komal Pandey**

GitHub:  
https://github.com/HiKomal11

---

##  License

This project is licensed under the MIT License.