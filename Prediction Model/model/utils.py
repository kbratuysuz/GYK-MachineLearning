import joblib
import pandas as pd
from datetime import datetime

def get_history(df: pd.DataFrame, customer_id: str, product_id: int, target_month: int):
    """
    Helper function to get previous purchase and rolling average of a customer's purchase history
    for a particular product until the target month.
    
    Parameters:
    - df: The data frame containing the customer purchase history.
    - customer_id: The ID of the customer.
    - product_id: The ID of the product.
    - target_month: The month for which we are calculating the history.
    
    Returns:
    - prev_purchase: The previous purchase for the product before the target month.
    - rolling_avg: The rolling average of purchases for the product before the target month.
    """
    # Filter and sort history
    history = df[
        (df['customer_id'] == customer_id) & 
        (df['product_id'] == product_id) &
        (df['order_month'] <= target_month)  # Only months before target
    ].sort_values('order_month')
    
    rolling_avg = history['has_purchased'].rolling(3, min_periods=1).mean().iloc[-1] 
    prev_purchase = history[history['order_month'] < target_month]['has_purchased'].iloc[-1] \
                    if any(history['order_month'] < target_month) else 0
    
    if not history.empty:
        return prev_purchase, rolling_avg
    return 0, 0  # Or appropriate default

def load_model(path: str):
    return joblib.load(path)

def make_prediction(model, product_id: int, date: str, customer_id: int):
    # Örnek veri hazırlama
    input_df = pd.DataFrame([{
        "customer_id": customer_id,
        "product_id": product_id,
        "order_month": pd.to_datetime(date),
    }])
    return model.predict(input_df)[0]

def retrain_model():
    # Burada veri yükleme ve yeniden eğitme yapılabilir
    # Eğitilen model tekrar kaydedilir
    pass
