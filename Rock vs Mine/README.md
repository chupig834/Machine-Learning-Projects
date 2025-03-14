# Sonar Rock vs. Mine Prediction Using Logistic Regression

## Project Overview
This project uses Logistic Regression to classify sonar signals as either originating from a rock or a mine based on sonar data features.

## Dataset Description
The Sonar dataset contains 208 samples with 60 numerical attributes each, representing sonar signals. The target is labeled:
- `M`: Mine
- `R`: Rock

## Technologies & Libraries
- Python
- Pandas (Data management)
- NumPy (Numerical calculations)
- Scikit-learn (Machine learning algorithms and evaluation)

## Model & Methodology
- **Algorithm**: Logistic Regression
- **Train-Test Split**: 90% training data, 10% testing data (stratified)

## Performance Metrics
- **Training Accuracy**: ~78.7%
- **Testing Accuracy**: ~76.2%

## Getting Started
### Installation
Install dependencies:
```bash
pip install numpy pandas scikit-learn
```

### Running the Project
Clone the repository:
```bash
git clone <repo_url>
```

Run the notebook using:
```bash
jupyter notebook
```

## Example Prediction
Use the model for predicting a single instance:
```python
input_data = (0.1150,0.1163,0.0866,0.0358,...,0.0166,0.0099) # 60 features
input_data_numpy = np.asarray(input_data)
input_data_reshaped = input_data_numpy.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
    print('The object detected is a Rock')
else:
    print('The object is a Mine')
```

## Future Improvements
- Test additional classification algorithms (Random Forest, SVM, etc.)
- Hyperparameter tuning to enhance accuracy
- Deploy the classifier as an application or API

## License
MIT

## Contact
Jerry Chu
- [LinkedIn](your-linkedin-url)
- [Email](mailto:your-email@example.com)

