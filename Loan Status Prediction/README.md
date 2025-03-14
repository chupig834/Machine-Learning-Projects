# Loan Approval Prediction using SVM

This project uses a Support Vector Machine (SVM) model with a linear kernel to predict loan approval based on applicant information.

## Dataset

The dataset `loandata.csv` includes information about loan applicants, such as income, marital status, education, credit history, and more.

### Columns:
- `Loan_ID`: Unique identifier for each loan application
- `Gender`: Gender of the applicant (Male/Female)
- `Married`: Marital status (Yes/No)
- `Dependents`: Number of dependents (0, 1, 2, 3+)
- `Education`: Education level (Graduate/Not Graduate)
- `Self_Employed`: Employment status (Yes/No)
- `ApplicantIncome`: Income of the applicant
- `CoapplicantIncome`: Income of the co-applicant
- `LoanAmount`: Amount of loan requested
- `Loan_Amount_Term`: Term of loan in months
- `Credit_History`: Credit history (1 indicates a positive credit history)
- `Property_Area`: Area where property is located (Urban, Semiurban, Rural)
- `Loan_Status`: Loan approval status (Y/N)

## Project Overview:
This project aims to predict loan approval using a supervised machine learning model (Support Vector Machine). The data is cleaned, processed, and encoded appropriately before training and evaluating the SVM model.

## Setup
### Dependencies
Ensure the following Python libraries are installed:

```bash
pip install numpy pandas seaborn scikit-learn
```

### Running the Code
Run the provided notebook (`.ipynb`) in Google Colab or any local Jupyter environment:

- Load the dataset (`loandata.csv`).
- Follow the preprocessing steps as outlined in the notebook.
- Train the SVM model.
- Evaluate the accuracy of the model.

## Data Processing Steps:
- Loaded the dataset and handled missing values by dropping rows with missing data.
- Encoded categorical variables numerically.
- Split the dataset into training and testing subsets (90% training, 10% testing).

## Model
### Support Vector Machine (SVM)
- **Kernel:** Linear
- **Accuracy:**
  - Training Data: ~79.86%
  - Testing Data: ~ similar performance (around 80%)

## Visualization
Visualizations include:
- Education vs. Loan Status
- Marital Status vs. Loan Acceptance Ratio

## Usage
You can input custom data points and use the trained SVM model to predict loan approval status.

## Contact
If you have questions or issues, please open an issue on GitHub or contact me directly.

---
**Happy Predicting! 🎯**

