import sqlite3
import pandas as pd

# Connect or create SQLite database
conn = sqlite3.connect("bookstore.db")

# Load CSVs into tables
cust_order = pd.read_csv("cust_order.csv")
cust_order.to_sql("cust_order", conn, if_exists="replace", index=False)

order_line = pd.read_csv("order_line.csv")
order_line.to_sql("order_line", conn, if_exists="replace", index=False)

# Optional: Churn predictions (can be empty initially)
churn_preds = pd.DataFrame(columns=["customer_id", "prediction", "prediction_date"])
churn_preds.to_sql("churn_predictions", conn, if_exists="replace", index=False)

conn.close()
print("✅ bookstore.db created successfully!")
