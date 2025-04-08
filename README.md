# GYK AI - Pair 1
- Merve Ceylan
- Burçin Acar
- Kübra Tüysüz Aksu

# Northwind Customer-Product Purchase Prediction Model

This project includes a machine learning model that predicts whether customers will purchase specific products using the Northwind database, along with a FastAPI-based web service that serves this model.

## Project Description

This system predicts the likelihood of a customer purchasing a specific product in the future by analyzing past purchasing behaviors. The model is trained using an XGBoost classifier and leverages the following features:

- Customer and product IDs
- Monthly seasonality (sine and cosine transformations)
- Customer-product interaction
- Previous month purchase behavior
- 3-month rolling average purchase behavior

## Requirements

To run this project, you'll need the following components:

- Python 3.8 or higher
- PostgreSQL database (with Northwind database installed)
- The following Python packages:
  - fastapi
  - uvicorn
  - pandas
  - numpy
  - scikit-learn
  - imblearn
  - xgboost
  - sqlalchemy
  - psycopg2
  - joblib
  - matplotlib 

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kbratuysuz/GYK-MachineLearning.git
cd GYK-MachineLearning/Prediction Model
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Create and import your Northwind database in PostgreSQL.

4. Update your database connection information in the `prediction-model.py` file and API service (`main.py`):

```python
DATABASE_URL = "postgresql+psycopg2://username:password@localhost:5432/NorthwindDB"
```

## Training the Model

### First Run (Model Training)

For the first run, you need to uncomment the parameter optimization section in the `prediction-model.py` file and comment out the model loading line:

1. Comment out this line:
```python
# Load the model later (ikinci çalıştırmada)
# model = joblib.load('model.joblib')
```

2. Uncomment the parameter optimization section:
```python
## PARAMETER OPTIMIZATION FOR THE MODEL
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

xgb = XGBClassifier(objective='binary:logistic',scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),random_state=156,enable_categorical=True)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                          cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
#grid_search.fit(X_train, y_train,eval_set=[(X_test, y_test)])
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)
best_score = grid_search.best_score_
print("Best ROC AUC score: ", best_score)

# To see all results in a DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)
print("\nAll results:")
print(results_df[['params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score'))

model = grid_search.best_estimator_ #save this model!!!

# Save the model
joblib.dump(model, 'model.joblib')  # or 'model.pkl'
```

3. Run the script to train and save the model:

```bash
python prediction-model.py
```

### Subsequent Runs (Using Saved Model)

After the first run, revert the changes:

1. Comment out the parameter optimization section
2. Uncomment the model loading line:
```python
# Load the model later 
model = joblib.load('model.joblib')
```

The training process:
- Pulls customer-product-month based purchase data from the Northwind database
- Processes this data and creates features
- Balances the dataset (using random undersampling method)
- Trains and saves the XGBoost model with optimized parameters
- Evaluates model performance and creates visualizations

## Running the API Service

To make the model accessible via API:

```bash
uvicorn main:app --reload
```

The service will run at http://localhost:8000 by default.

## API Reference

Once the API is running, you can access the swagger API documentation at: http://localhost:8000/docs

Or you can access the detailed API Reference at: https://kbratuysuz.github.io/GYK-MachineLearning/

### Main Endpoints

- `GET /`: Check API status
- `GET /products`: List all products
- `GET /categories`: List all categories
- `GET /sales_summary`: Provides a sales summary
- `POST /predict`: Makes a prediction for a specific customer-product-month combination
- `POST /retrain`: Retrains the model

### Example Prediction Request

```json
{
  "customer_id": "ALFKI",
  "product_id": 11,
  "order_month": 3
}
```

## Docker Support

The application is available as a Docker container on Docker Hub. This is the easiest way to run the application without installing any dependencies.

### Running with Docker

1. Pull the Docker image from Docker Hub:

```bash
docker pull brcnacar/pair1
```

2. Run the Docker container:

```bash
docker run -p 8000:8000 brcnacar/pair1
```

This will start the API service on port 8000. You can access it at http://localhost:8000.

That's it! The application is now running in a container with all dependencies included.

## Model Performance

### Feature Importance
![WhatsApp Image 2025-04-08 at 14 15 40](https://github.com/user-attachments/assets/dcc5932b-4101-45c1-a584-b7855de18439)

### ROC Curve
![WhatsApp Image 2025-04-08 at 14 15 40 (2)](https://github.com/user-attachments/assets/d623b260-8c68-47d6-9a60-d1ac0767894e)

### First Tree in XGBoost Model
![WhatsApp Image 2025-04-08 at 14 15 40 (1)](https://github.com/user-attachments/assets/cda921a4-9480-473b-a96e-f34eec3e7729)

### Classification Report
![WhatsApp Image 2025-04-08 at 14 15 40 (3)](https://github.com/user-attachments/assets/b78fe2ed-355e-473a-bc04-b1e1052f0150)
