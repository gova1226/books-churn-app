import sqlite3

# Connect to or create the SQLite database
conn = sqlite3.connect("bookstore.db")
cursor = conn.cursor()

# Create cust_order table
cursor.execute("""
CREATE TABLE IF NOT EXISTS cust_order (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date TEXT
)
""")

# Create order_line table
cursor.execute("""
CREATE TABLE IF NOT EXISTS order_line (
    order_line_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    price REAL,
    FOREIGN KEY(order_id) REFERENCES cust_order(order_id)
)
""")

# Create churn_predictions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS churn_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    prediction INTEGER,
    prediction_date TEXT
)
""")

conn.commit()
conn.close()

print("✅ bookstore.db created with required tables.")