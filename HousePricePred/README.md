# California Housing Price Prediction using XGBoost

## Project Overview
This project leverages the XGBoost regression algorithm to predict median house prices in California. The dataset used is the California Housing dataset available from Scikit-learn.

## Dataset Description
The dataset contains 20,640 entries with the following attributes:
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average occupancy (number of household members)
- `Latitude`: Geographic latitude
- `Longitude`: Geographic longitude
- `price`: Median house value ($100,000s)

## Technologies & Libraries
- Python
- Pandas (Data handling)
- NumPy (Numerical computing)
- Scikit-learn (Dataset and data splitting)
- XGBoost (Model training)
- Matplotlib & Seaborn (Data visualization)

## Model & Methodology
- **Algorithm**: XGBoost Regression
- **Train-Test Split**: 80% training data, 20% testing data

## Model Performance
| Metric                      | Training Set  | Testing Set  |
|-----------------------------|---------------|--------------|
| R-squared error             | ~94.5%        | ~83.3%       |
| Mean Absolute Error (MAE)   | ~0.192        | ~0.305       |

## Getting Started
### Installation
Install necessary libraries:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Running the Project
Clone the repository:
```bash
git clone <repo_url>
```

Run the notebook using Jupyter:
```bash
jupyter notebook
```

## Visualization
The project includes visualizations to compare actual vs. predicted house prices.
```python
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()
```

## Future Enhancements
- Hyperparameter tuning using GridSearchCV
- Experiment with additional regression algorithms
- Feature engineering to improve model accuracy
- Deploying the model using web applications

## License
MIT

## Contact
Jerry Chu
- [LinkedIn](your-linkedin-url)
- [Email](mailto:your-email@example.com)

