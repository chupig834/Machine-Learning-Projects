# Diabetes Prediction Using Support Vector Machine (SVM)

## Project Overview
This project aims to predict diabetes in females using machine learning techniques, specifically a Support Vector Machine (SVM) with a linear kernel. The dataset used is included in the repository.
## Dataset
The dataset contains 768 records with the following attributes:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skinfold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age in years
- `Outcome`: Diabetes outcome (0: Non-Diabetic, 1: Diabetic)

## Technologies & Libraries
- Python
- Pandas (Data manipulation and analysis)
- NumPy (Numerical operations)
- Scikit-learn (Machine learning models)

## Model & Methodology
- **Algorithm**: Support Vector Machine (Linear Kernel)
- **Data Standardization**: Scikit-learn's StandardScaler
- **Train-Test Split**: 80% training, 20% testing (with stratification)

## Performance
- **Training Accuracy**: ~78.7%
- **Testing Accuracy**: ~77.3%

## Getting Started
### Installation
Install dependencies:
```bash
pip install numpy pandas scikit-learn
```

### Usage
Clone the repository:
```bash
git clone <repo_url>
```

Run the notebook with:
```bash
jupyter notebook
```

## Example Prediction
```python
input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_numpy = np.asarray(input_data)
input_data_reshape = input_data_numpy.reshape(1, -1)
std_data = scaler.transform(input_data_reshape)

prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person does not have diabetes')
else:
    print('The person has diabetes')
```

## Future Improvements
- Experiment with other kernels (Polynomial, RBF)
- Hyperparameter tuning (GridSearchCV)
- Implement feature selection methods
- Deploy the model in a web-based or mobile application

## License
MIT

