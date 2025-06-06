# Fake News Detection using Machine Learning

## Overview
This project detects potentially unreliable (fake) news articles using Natural Language Processing (NLP) techniques and Logistic Regression.

## Dataset
- **train.csv** contains 20,800 articles labeled as:
  - `1`: Unreliable (Fake)
  - `0`: Reliable (Real)
- **train.csv** - https://www.kaggle.com/c/fake-news/data?select=train.csv

## Technologies & Libraries
- **Python**
- **Pandas** (Data analysis)
- **NumPy** (Numerical operations)
- **NLTK** (Natural Language Toolkit)
- **Scikit-learn** (Machine learning algorithms)

## Data Preprocessing Steps
- **Missing Values**: Identified and handled missing values.
- **Feature Engineering**: Combined 'author' and 'title' into a single 'content' feature.
- **Stemming**: Reduced words to their root form to simplify the text data.
- **Text Vectorization**: Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical vectors.

## Model
- **Algorithm**: Logistic Regression
- **Accuracy**:
  - Training Data: ~97.9%
  - Testing Data: ~97.8%

## Libraries Required
Install required libraries using:
```bash
pip install numpy pandas nltk scikit-learn
```

Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Running the Code
1. Clone this repository:
```bash
git clone <repo_url>
```

2. Navigate to the project directory and run the notebook:
```bash
jupyter notebook
```

## Usage Example
```python
# Predict a sample from test data
X_new = X_test[3]
prediction = model.predict(X_new)
if prediction[0] == 0:
    print('The news is real')
else:
    print('The news is fake')
```

## Future Work
- Explore additional classifiers like Random Forest, SVM, or neural networks.
- Deploy as a web app for public use.
- Integrate additional features (article images, metadata, etc.).

## License
MIT

