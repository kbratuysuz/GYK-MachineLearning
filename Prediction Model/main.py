from fastapi import FastAPI, HTTPException,Response
from pydantic import BaseModel
import joblib
from model.utils import load_model, make_prediction, retrain_model, get_history
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from flask import jsonify

app = FastAPI(title="ML Prediction API")

## LINKING THE NORTHWIND DATABASE
DATABASE_URL = "postgresql://burcinacar:sexed6925--@localhost:5432/deneme"

# Create an engine
engine = create_engine(DATABASE_URL, echo=True)

df_filtered = pd.read_csv('data/data_filtered.csv')  # Loading your dataset
df_filtered = df_filtered.sort_values(['customer_id', 'product_id', 'order_month'])

# Load the saved encoders
interaction_encoder = joblib.load('model/interaction_encoder.pkl')
customer_encoder = joblib.load('model/customer_encoder.pkl')
product_encoder = joblib.load('model/product_encoder.pkl')

# Ana sayfa (root) endpoint
@app.get("/")
def read_root():
    try:
        return {"status": "up", "message": "API is ready!"}
    except Exception as e:
        return Response(
            status_code=503, 
            content=json.dumps({
                "status": "down",
                "error": "ECONNREFUSED",
                "message": "API service is not available",
                "details": str(e)
            }),
            media_type="application/json"
        )

# Veri modelleri
class PredictRequest(BaseModel):
    customer_id: str
    product_id: int
    order_month: int

@app.get("/products")
def get_products():
    try:
        getdf = pd.read_sql("""SELECT * FROM products""",engine)
        data = getdf.to_dict(orient="records")
        return data
    except Exception as e:
        return Response(
            status_code=404, 
            content=json.dumps({
                "message": "Error while getting products.",
                "details": str(e)
            }),
            media_type="application/json"
        )

@app.get("/categories")
def get_categories():
    try:
        getdfcat = pd.read_sql("""SELECT * FROM categories""",engine)
        datac = getdfcat.to_dict(orient="records")
        return datac
    except Exception as e:
        return Response(
            status_code=404, 
            content=json.dumps({
                "message": "Error while getting products.",
                "details": str(e)
            }),
            media_type="application/json"
        )


@app.post("/predict")
def predict(req: PredictRequest):
    model = load_model("model/model.joblib")
    customer_id = req.customer_id
    product_id = req.product_id
    order_month = req.order_month

    try:
    # Call get_history function to calculate prev_purchase and rolling_avg
        prev_purchase, rolling_avg = get_history(df_filtered, customer_id, product_id, order_month)
    
    # Prepare the example DataFrame
        example = pd.DataFrame({'customer_id': [customer_id], 'product_id': [product_id], 'order_month': [order_month]})
    
    # Add previous month purchase and rolling average columns
        example['prev_month_purchase'] = prev_purchase
        example['rolling_3month'] = rolling_avg

    # Encode categoricals
        example['customer_product_interaction'] = example['customer_id'].astype(str) + "_" + example['product_id'].astype(str)
        example["customer_product_interaction"] = interaction_encoder.transform(example["customer_product_interaction"])
        example["customer_id"] = customer_encoder.transform(example["customer_id"])
        example["product_id"] = product_encoder.transform(example["product_id"])

    # Set 1 in the dummy columns based on the example
        example["month_sin"] = np.sin(2 * np.pi * example["order_month"].astype(int) / 12)
        example["month_cos"] = np.cos(2 * np.pi * example["order_month"].astype(int) / 12)
        example.drop(columns="order_month", inplace=True)

    # Reorder columns to match the model features
        example_colorder = example[model.feature_names_in_]
    
    # Make prediction
        prediction = model.predict(example_colorder)
        prediction_value = int(prediction[0])
        if prediction_value == 0:
            result = "Won't buy"
        elif prediction_value == 1:
            result = "Will buy"
        else:
            result = "Unknown" 
    
      # Ensure this is a simple int
        return result
    except Exception as e:
        return "This customer never purchased this product."

@app.post("/retrain")
def retrain():
    try:
        retrain_model()
        return {"message": "Model retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/sales_summary")
def sales_summary():
    try:
        getstats = pd.read_sql("""WITH product_sales AS (
            SELECT 
                p.product_name,
                SUM(od.quantity) AS toplam
            FROM 
                order_details od
            JOIN products p ON od.product_id = p.product_id
            GROUP BY 
                p.product_name
            ),
            most_sold AS (
                SELECT 
                    product_name,
                    toplam
                FROM 
                    product_sales
                ORDER BY 
                    toplam DESC
                LIMIT 1
            ),
            totals AS (
                SELECT
                    SUM(od.quantity * od.unit_price) AS total_sales,
                    SUM(od.quantity) AS total_items
                FROM 
                    order_details od
            )

            SELECT
                totals.total_sales AS total_sales_amount,
                totals.total_items AS total_items_sold,
                most_sold.product_name AS most_sold_product,
                most_sold.toplam AS most_sold_quantity
            FROM 
                totals, most_sold;""",engine)
        data = getstats.to_dict(orient="records")
        return data
    except Exception as e:
        return Response(
            status_code=404, 
            content=json.dumps({
                "message": "Error while getting sales summary.",
                "details": str(e)
            }),
            media_type="application/json"
        )