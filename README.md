---
title: Fake News Detection Using Machine Learning
emoji: 📰
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Fake News Detection using Machine Learning and NLP

## Final Year Project - Implementation Guide

### Project Overview
This project implements a fake news detection system using **Logistic Regression** combined with **Natural Language Processing (NLP)** techniques. The system analyzes news articles and classifies them as either reliable or fake news.

---

## Features

- **Text Preprocessing**: Comprehensive NLP pipeline including:
  - Lowercase conversion
  - URL and mention removal
  - Punctuation and number removal
  - Stopword filtering
  - Stemming using Porter Stemmer

- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization

- **Machine Learning**: Logistic Regression classifier

- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix visualization

- **Single Article Prediction**: Test individual news articles

---

## Installation

### 1. Install Python
Ensure you have Python 3.8 or higher installed.

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

---

## Dataset Sources

For your final year project, you can use one of these popular fake news datasets:

### **Recommended Datasets:**

1. **Kaggle Fake News Dataset**
   - **Link**: https://www.kaggle.com/c/fake-news/data
   - **Size**: ~20,000 articles
   - **Format**: CSV with columns: id, title, author, text, label
   - **Download**: Requires Kaggle account

2. **ISOT Fake News Dataset**
   - **Link**: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
   - **Size**: ~44,000 articles
   - **Types**: Political news (real and fake)
   - **Format**: Separate CSV files for real and fake news

3. **LIAR Dataset**
   - **Link**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
   - **Size**: 12,800 statements
   - **Labels**: 6 fine-grained labels (true, mostly-true, half-true, barely-true, false, pants-fire)

4. **FakeNewsNet Dataset**
   - **Link**: https://github.com/KaiDMML/FakeNewsNet
   - **Features**: Includes social context and user engagements
   - **Format**: JSON files

---

## Dataset Format

Your CSV file should have at least these columns:

```
text,label
"News article text here...",0
"Another news article...",1
```

Where:
- `text`: The news article content
- `label`: 0 for reliable/real news, 1 for fake news

### Example Dataset Structure:

```csv
text,label
"The president announced new policies during a press conference today",0
"SHOCKING! You won't believe this miracle cure that doctors don't want you to know!",1
```

---

## How to Download Datasets

### Method 1: Kaggle Dataset

1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/c/fake-news/data
3. Click "Download All"
4. Extract the files (train.csv, test.csv)
5. Use train.csv for training

### Method 2: Manual Download

1. Visit the dataset link
2. Download the CSV file
3. Place it in your project directory
4. Update the filepath in the code

---

## Project Structure

```
FND Gemini/
├── .env                    # Environment variables (API keys)
├── README.md               # Project guide
├── app.py                  # Main Flask application
├── model.pkl               # Trained text model
├── vectorizer.pkl          # Trained TF-IDF vectorizer
├── requirements.txt        # Dependencies
├── run_app.bat             # Batch script to run the app
├── train_and_save_model.py # Script to retrain the model
├── train.csv               # Training dataset
├── static/                 # Web assets and image uploads
└── templates/              # HTML templates
```

---

## Usage

### 1. Run the Web Application
Launch the Flask server:
```bash
python app.py
```
Or use the provided batch script:
```bash
run_app.bat
```

### 2. Retrain the Model
If you update `train.csv`, retrain the model by running:
```bash
python train_and_save_model.py
```
This will regenerate `model.pkl` and `vectorizer.pkl`.

---

## Code Workflow

1. **Data Loading**: Load CSV dataset
2. **Preprocessing**: Clean and process text data
3. **Feature Extraction**: Convert text to TF-IDF vectors
4. **Train-Test Split**: 80% training, 20% testing
5. **Model Training**: Train Logistic Regression
6. **Prediction**: Make predictions on test set
7. **Evaluation**: Calculate metrics and visualize results

---

## Output Example

```
==================================================
FAKE NEWS DETECTION SYSTEM
Using Logistic Regression & NLP
==================================================

Loading dataset...
Dataset shape: (20000, 2)

Training samples: 16000
Testing samples: 4000

Training the model...
Preprocessing training data...
Vectorizing text...
Training Logistic Regression...
Training completed!

==================================================
MODEL EVALUATION RESULTS
==================================================

Accuracy: 0.9245
Precision: 0.9234
Recall: 0.9245
F1-Score: 0.9238

--------------------------------------------------
DETAILED CLASSIFICATION REPORT
--------------------------------------------------
              precision    recall  f1-score   support

    Reliable       0.91      0.94      0.93      2000
        Fake       0.94      0.91      0.92      2000

    accuracy                           0.92      4000
   macro avg       0.92      0.92      0.92      4000
weighted avg       0.92      0.92      0.92      4000

--------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------
[[1880  120]
 [ 182 1818]]
```

---

## Customization Options

### 1. Adjust Model Parameters

```python
# Increase iterations for better convergence
self.model = LogisticRegression(max_iter=2000, random_state=42)

# Add regularization
self.model = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
```

### 2. Change TF-IDF Features

```python
# Increase feature count
self.vectorizer = TfidfVectorizer(max_features=10000)

# Add n-grams
self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
```

### 3. Try Different Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # 70-30 split
)
```

---

## Project Report Sections

For your final year project report, include:

1. **Abstract**: Brief overview of the project
2. **Introduction**: Problem statement and objectives
3. **Literature Review**: Related work in fake news detection
4. **Methodology**: 
   - Data collection and preprocessing
   - Feature extraction (TF-IDF)
   - Logistic Regression algorithm
5. **Implementation**: Code explanation
6. **Results**: Performance metrics and visualizations
7. **Conclusion**: Findings and future work
8. **References**: Citations

---

## Performance Tips

- Use larger datasets for better accuracy (20,000+ articles)
- Experiment with different preprocessing techniques
- Try combining title and text columns
- Consider adding more features (author, source, etc.)
- Cross-validation for robust evaluation

---

## Troubleshooting

### Issue: Low Accuracy
**Solution**: 
- Use more training data
- Increase TF-IDF features
- Try different preprocessing

### Issue: Memory Error
**Solution**: 
- Reduce max_features in TfidfVectorizer
- Use smaller dataset for testing
- Process data in batches

### Issue: NLTK Download Error
**Solution**: 
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Future Enhancements

- Implement other ML algorithms (Random Forest, SVM, Neural Networks)
- Add deep learning models (LSTM, BERT)
- Create web interface using Flask/Streamlit
- Real-time news verification
- Multi-label classification (degrees of fakeness)
- Integration with news APIs

---

## License

This project is for educational purposes (Final Year Project).

---

## Contact & Support

For questions or issues with this implementation, refer to:
- scikit-learn documentation: https://scikit-learn.org/
- NLTK documentation: https://www.nltk.org/
- Pandas documentation: https://pandas.pydata.org/

---

## Acknowledgments

- Dataset providers (Kaggle, ISOT, etc.)
- scikit-learn library
- NLTK library

---

**Good luck with your final year project! 🎓**
