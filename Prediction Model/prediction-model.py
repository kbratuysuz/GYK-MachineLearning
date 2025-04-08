## MODULES
from imblearn.under_sampling import RandomUnderSampler
from sqlalchemy import create_engine # psycopg2 is the connection bridge between sqlalchemy and postgresql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import joblib 
from xgboost import plot_tree


## LINKING THE NORTHWIND DATABASE
DATABASE_URL = "postgresql+psycopg2://postgres:1@localhost:5432/GYK2Northwind"

# Create an engine
engine = create_engine(DATABASE_URL, echo=True)

# Read relevant data
df = pd.read_sql("""WITH months AS (
    SELECT generate_series(1, 12) AS order_month
),
customer_product_months AS (
    SELECT 
        c.customer_id,
        p.product_id,
        m.order_month
    FROM customers c
    CROSS JOIN products p
    CROSS JOIN months m
),
purchases AS (
    SELECT 
        o.customer_id,
        od.product_id,
        EXTRACT(MONTH FROM o.order_date)::int AS order_month
    FROM orders o
    JOIN order_details od ON o.order_id = od.order_id
    WHERE o.order_date IS NOT NULL
    GROUP BY o.customer_id, od.product_id, order_month
),
final AS (
    SELECT 
        cpm.customer_id,
        cpm.product_id,
        LPAD(cpm.order_month::text, 2, '0') AS order_month,
        CASE 
            WHEN p.customer_id IS NOT NULL THEN 1 
            ELSE 0 
        END AS has_purchased
    FROM customer_product_months cpm
    LEFT JOIN purchases p
        ON cpm.customer_id = p.customer_id 
       AND cpm.product_id = p.product_id 
       AND cpm.order_month = p.order_month
)
SELECT * FROM final
ORDER BY customer_id, product_id, order_month;""",engine)

df["order_month"]=df["order_month"].astype(int)
df["product_id"]=df["product_id"].astype(object)

## INFORMATION ON DATA
class_counts = df['has_purchased'].value_counts()
print("Number of elements in each class:")
print(class_counts)
df.groupby(["customer_id","product_id"])["has_purchased"].sum().plot(kind="bar")
# plt.show()

## REMOVING CUSTOMER-PRODUCT ID PAIRS IF NO PURCHASE DATA IS GIVEN FOR ANY MONTH
df_filtered = df[df.groupby(["customer_id", "product_id"])["has_purchased"].transform('sum') > 0]
#print(df_filtered.head(45))
class_counts2 = df_filtered['has_purchased'].value_counts()
print("Number of elements in each filtered class:")
print(class_counts2)
del df
df=df_filtered
df.to_csv("data_filtered.csv", index=False)


## LABEL ENCODING FOR CATEGORICAL ATTRIBUTES
customer_encoder = LabelEncoder()
product_encoder = LabelEncoder()
interaction_encoder = LabelEncoder()

df['customer_product_interaction'] = df['customer_id'].astype(str) + "_" + df['product_id'].astype(str)
df['customer_product_interaction'] = interaction_encoder.fit_transform(df['customer_product_interaction'])
joblib.dump(interaction_encoder, 'model/interaction_encoder.pkl')

df = df.sort_values(['customer_id', 'product_id', 'order_month'])
df['prev_month_purchase'] = df.groupby(['customer_id', 'product_id'])['has_purchased'].shift(1)
df['prev_month_purchase'].fillna(0, inplace=True)
df['rolling_3month'] = df.groupby(['customer_id', 'product_id'])['has_purchased'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Feature engineering
df["month_sin"] = np.sin(2 * np.pi * df["order_month"].astype(int) / 12)
df["month_cos"] = np.cos(2 * np.pi * df["order_month"].astype(int) / 12)

# Encode categoricals
df["customer_id"] = customer_encoder.fit_transform(df["customer_id"])
df["product_id"] = product_encoder.fit_transform(df["product_id"])
joblib.dump(customer_encoder, 'model/customer_encoder.pkl')
joblib.dump(product_encoder, 'model/product_encoder.pkl')

df.to_csv("data/data.csv", index=False)
print(df.loc[33])
print(df['customer_product_interaction'].min)

# Split data
X = df[["customer_id", "product_id", "month_sin", "month_cos","customer_product_interaction","prev_month_purchase","rolling_3month"]]
y = df["has_purchased"]
X['customer_id'] = X['customer_id'].astype('category')
X['product_id'] = X['product_id'].astype('category')
X["customer_product_interaction"]=X["customer_product_interaction"].astype('category')

## BALANCING DATASET
rus = RandomUnderSampler(random_state=42)
Xx, yx = rus.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(Xx, yx, test_size=0.2,random_state=156,stratify=yx)

## PARAMETER OPTIMIZATION FOR THE MODEL
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'min_child_weight': [1, 3, 5]
# }

# xgb = XGBClassifier(objective='binary:logistic',scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),random_state=156,enable_categorical=True)
# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
#                           cv=5, scoring='roc_auc', n_jobs=-1)
# grid_search.fit(X_train, y_train)
# #grid_search.fit(X_train, y_train,eval_set=[(X_test, y_test)])
# best_params = grid_search.best_params_
# print("Best parameters found: ", best_params)
# best_score = grid_search.best_score_
# print("Best ROC AUC score: ", best_score)

# # To see all results in a DataFrame
# results_df = pd.DataFrame(grid_search.cv_results_)
# print("\nAll results:")
# print(results_df[['params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score'))

# model = grid_search.best_estimator_ #save this model!!!

# # Save the model
# joblib.dump(model, 'model.joblib')  # or 'model.pkl'


# Load the model later (ikinci çalıştırmada)
model = joblib.load('model.joblib')

# Example
#example = pd.DataFrame({'customer_id': ['ALFKI'], 'product_id': [3], 'order_month': [10]})
#example = pd.DataFrame({'customer_id': ["ANTON"], 'product_id': [11], 'order_month': [11]})
example = pd.DataFrame({'customer_id': ["WELLI"], 'product_id': [70], 'order_month': [2]})
# when order month is small it takes only 2 months like here, does not take the previous december, or before
def get_history(df, customer_id, product_id, target_month):
    # Filter and sort history
    history = df[
        (df['customer_id'] == customer_id[0]) & 
        (df['product_id'] == product_id[0]) &
        (df['order_month'] <= target_month[0])  # Only months before target
    ].sort_values('order_month')
    print(history)
    rolling_avg = history['has_purchased'].rolling(3, min_periods=1).mean().iloc[-1] 
    prev_purchase = history[history['order_month'] < target_month[0]]['has_purchased'].iloc[-1] \
                    if any(history['order_month'] < target_month[0]) else 0
    if not history.empty:
        return prev_purchase,rolling_avg
    return 0,0  # or appropriate default


# Create a new DataFrame initialized with zeros
df_filtered = df_filtered.sort_values(['customer_id', 'product_id', 'order_month'])

example['prev_month_purchase'],example['rolling_3month']=get_history(df_filtered,example['customer_id'],example['product_id'],example['order_month'])

# Encode categoricals
example['customer_product_interaction'] = example['customer_id'].astype(str) + "_" + example['product_id'].astype(str)
example["customer_product_interaction"]=interaction_encoder.transform(example["customer_product_interaction"])
example["customer_id"] = customer_encoder.transform(example["customer_id"])
example["product_id"] = product_encoder.transform(example["product_id"])

# Set 1 in the dummy columns based on the example
example["month_sin"] = np.sin(2 * np.pi * example["order_month"].astype(int) / 12)
example["month_cos"] = np.cos(2 * np.pi * example["order_month"].astype(int) / 12)
example.drop(columns="order_month",inplace=True)

example_colorder = example[model.feature_names_in_]
predictions = model.predict(example_colorder)  # Now it works!
print(predictions)
print(example)
print(X_train)


# After your model training code...

# Get predicted probabilities for the positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot the first tree (index 0)
# plt.figure(figsize=(100, 60))
# plot_tree(model, num_trees=0,max_depth=2)  # Try different tree indices
# plt.title("First Tree in XGBoost Model")
# plt.savefig('xgb2_tree.png', dpi=4300, bbox_inches='tight')  # High DPI
# plt.show()

## evaluation

y_pred = model.predict(X_test)

# Classification report (already in your code)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Precision-Recall AUC
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Özellik isimleri ve importanceları
feature_importance = model.feature_importances_
features = X.columns

# DataFrame'e çevir
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Görselleştir
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Önem Skoru')
plt.ylabel('Özellik')
plt.title('Özellik Önem Sıralaması (XGBoost)')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()