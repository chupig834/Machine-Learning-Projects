# Wine Quality Prediction

## Overview
This project utilizes machine learning techniques to predict the quality of red wine based on its physicochemical properties. The dataset is sourced from the UCI Machine Learning Repository and contains 1,599 observations with 12 attributes.

A **Random Forest Classifier** is trained on the dataset to classify wines as either good quality (quality >= 7) or bad quality (quality < 7).

## Dataset
The dataset used for this project is `winequality-red.csv`, which contains the following features:

- **Fixed Acidity**
- **Volatile Acidity**
- **Citric Acid**
- **Residual Sugar**
- **Chlorides**
- **Free Sulfur Dioxide**
- **Total Sulfur Dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**
- **Quality (Target Variable)**

## Installation
To run this project, ensure you have Python installed along with the required dependencies. You can install them using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
### 1. Import Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

### 2. Load and Explore the Data
```python
wine_dataset = pd.read_csv('winequality-red.csv')
print(wine_dataset.head())
print(wine_dataset.isnull().sum())
```

### 3. Visualize Data
```python
sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.show()
```

### 4. Correlation Analysis
```python
correlation = wine_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, cmap='Blues')
plt.show()
```

### 5. Data Preparation
```python
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
```

### 6. Model Training
```python
model = RandomForestClassifier()
model.fit(X_train, Y_train)
```

### 7. Model Evaluation
```python
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy: ", test_data_accuracy)
```

### 8. Making Predictions
```python
input_data = (6.3, 0.39, 0.16, 1.4, 0.08, 11.0, 23.0, 0.9955, 3.34, 0.56, 9.3)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

if prediction == 1:
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")
```

## Results
- **Testing Accuracy:** ~93.1%
- **Key Features Impacting Quality:**
  - **Positively correlated:** Alcohol, Sulphates
  - **Negatively correlated:** Volatile Acidity

## Future Improvements
- Try different models (e.g., SVM, Gradient Boosting)
- Hyperparameter tuning for better accuracy
- Use a larger dataset for better generalization

## License
This project is for educational purposes only and follows open-source licensing norms.

## Author
Your Name (Optional)

