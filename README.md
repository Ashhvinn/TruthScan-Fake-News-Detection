# **TruthScan Fake News Detection**

This project implements **TruthScan**, a fake news detection system that leverages machine learning algorithms to identify and classify news articles as real or fake. The system analyzes various features of news articles, including text content, writing style, and external data sources, to determine the authenticity of the information.

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation Guide](#installation-guide)
5. [Usage](#usage)
6. [Model Details](#model-details)
7. [Dataset](#dataset)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Challenges and Future Enhancements](#challenges-and-future-enhancements)

---

## **Introduction**

**TruthScan Fake News Detection** is designed to help users identify fake news articles using machine learning and natural language processing techniques. It can be integrated into various applications to automatically flag misleading or fabricated news. The project combines deep learning models with feature extraction methods to improve detection accuracy.

---

## **Features**

- **Text Classification**: Classifies news articles as real or fake based on content analysis.
- **Feature Extraction**: Uses NLP techniques such as tokenization, named entity recognition (NER), and sentiment analysis.
- **Preprocessing Pipeline**: Implements an efficient pipeline for data cleaning, tokenization, and feature extraction.
- **Model Training**: Trains machine learning models using labeled news datasets to classify articles.
- **Web Interface**: Provides a user-friendly interface to upload and evaluate articles.

---

## **Prerequisites**

Ensure the following tools and libraries are installed:

- **Python 3.7+**
- **Scikit-learn**
- **TensorFlow / PyTorch**
- **NLTK / spaCy** (for text preprocessing)
- **Flask** (for web interface)
- **Pandas / NumPy** (for data manipulation)
- **Matplotlib** (for visualizations)

---

## **Installation Guide**

### **Clone the Repository**
```bash
git clone https://github.com/your_username/TruthScan-Fake-News-Detection.git
cd TruthScan-Fake-News-Detection
```
### **Install Dependencies**
```bash
pip install -r requirements.txt
```
### **Prepare the Dataset**
Place the dataset in the data/ directory
```bash
mkdir -p data
```
Move your labeled dataset (CSV or JSON format) into the data/ directory
```bash
echo "Dataset directory created. Please place your dataset files in the 'data/' directory."
```

---

## **Usage**

### **Training the Model**

To train the fake news detection model, use the preprocessed dataset by running the following command:
```bash
python train_model.py
```
This script will load the preprocessed data from the data/ directory, train a classification model (e.g., RandomForest, Logistic Regression, or deep learning models like LSTM or BERT), and save the trained model to a file for later use.

### **Evaluating the Model**
After training the model, you can evaluate its performance on a validation or test set using the following command:
```bash
python evaluate_model.py
```
This script will load the trained model, apply it to the test dataset, and output evaluation metrics such as accuracy, precision, recall, and F1-score to assess the model's performance in detecting fake news.

### **Launching the Web Interface**
To interact with the trained model via a web interface, you can launch the Flask web application using:
```bash
python app.py
```
This command will start a local server where you can upload news articles and get predictions about whether they are real or fake. You can access the web interface by navigating to http://127.0.0.1:5000/ in your browser.

---

## **Model Details**

### **Architecture**

The **TruthScan** Fake News Detection model leverages a variety of techniques for preprocessing, feature extraction, and classification. Below is a breakdown of the architecture and process used in the project:

#### **Preprocessing and Feature Extraction**

The text data is preprocessed using several Natural Language Processing (NLP) techniques to extract relevant features:

- **Tokenization**: The text is broken into tokens, which can be words or subwords, using tokenization techniques.
- **Named Entity Recognition (NER)**: This step identifies entities such as people, organizations, and locations in the news articles.
- **Sentiment Analysis**: The sentiment of each news article is analyzed, helping to capture the emotional tone of the content.

#### **Classification Models**

The project employs a combination of machine learning models for classifying articles as real or fake:

- **Logistic Regression**
- **Random Forest**
- **Deep Learning Models**: Models like **LSTM**, **BERT**, and **RoBERTa** are used to enhance classification accuracy by leveraging sequential patterns in text data.

### **Pipeline**

The overall pipeline consists of the following steps:

1. **Data Preprocessing**  
   - Cleaning and preparing text data for analysis.
   - Tokenizing text and performing additional NLP tasks such as NER and sentiment analysis.
   
2. **Feature Extraction**  
   - Extracting meaningful features from the text, such as word counts, named entities, and sentiment scores.

3. **Model Training**  
   - Training the classification models using libraries like **scikit-learn** (for traditional models) or **TensorFlow/Keras** (for deep learning models).

4. **Prediction**  
   - Using the trained model to predict whether a given article is real or fake based on its features.

### **Code Example**

Here’s an example code for training a **Random Forest** classifier with the news dataset:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('data/news_dataset.csv')

# Preprocessing and feature extraction
# (Implement necessary preprocessing here)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
```

---


## **Dataset**

The dataset used for training and evaluating the **TruthScan** Fake News Detection model consists of labeled news articles. Each article is labeled with a 'label' column indicating whether the article is **real** or **fake**.

### **Dataset Source**

[Insert link to dataset or location]

### **Format**

The dataset is available in either **CSV** or **JSON** format, with the following columns:

- **content**: The text of the news article.
- **label**: The classification of the article (either "real" or "fake").

### **Sample Data**

Here is a small sample of the dataset:

| content                                                   | label |
|-----------------------------------------------------------|-------|
| "President announces new policy on healthcare"            | real  |
| "Aliens found on Mars! Shocking discovery!"               | fake  |

Ensure the dataset is placed in the `data/` directory of the project for proper integration with the training and evaluation scripts.

---

## **Evaluation Metrics**

To evaluate the performance of the **TruthScan** Fake News Detection model, the following metrics are used:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision & Recall**: Evaluates the classification performance for real and fake news, especially useful when the dataset is imbalanced.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between them.
- **Confusion Matrix**: A detailed performance breakdown, showing the true positives, true negatives, false positives, and false negatives.

These metrics are crucial for assessing how well the model distinguishes between real and fake news.

---

## **Challenges and Future Enhancements**

### **Challenges**

- **Data Imbalance**: Fake news datasets may have fewer fake articles compared to real ones, leading to imbalanced class distribution.
- **Feature Extraction**: Extracting meaningful features from raw text, such as named entities, sentiment, and topic modeling, can be difficult and computationally expensive.
  
### **Future Enhancements**

- **Multilingual Support**: Extend the model to handle multiple languages for broader applicability.
- **Deep Learning Models**: Implement more advanced models like **BERT** and **RoBERTa** to achieve higher accuracy in fake news detection.
- **Web Scraping**: Automate the data collection process from online news sources to keep the dataset up-to-date and more comprehensive.

These improvements will help make the model more robust, adaptable, and capable of handling real-world challenges in fake news detection.

---






