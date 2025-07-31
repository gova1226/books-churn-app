import sqlite3
import pandas as pd

# Connect (or create) SQLite database
conn = sqlite3.connect("gravity_books.db")

# Load and insert CSVs into SQLite tables
cust_order = pd.read_csv("cust_order.csv")
cust_order.to_sql("cust_order", conn, if_exists="replace", index=False)

order_line = pd.read_csv("order_line.csv")
order_line.to_sql("order_line", conn, if_exists="replace", index=False)

# Optional: Add churn_predictions table with or without data
try:
    churn_preds = pd.read_csv("churn_predictions.csv")
    churn_preds.to_sql("churn_predictions", conn, if_exists="replace", index=False)
except FileNotFoundError:
    # Create empty table if no data exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS churn_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            prediction INTEGER,
            prediction_date TEXT
        );
    """)

conn.commit()
conn.close()
print("✅ bookstore.db created successfully!")
