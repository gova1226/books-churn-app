import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import datetime
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# ✅ Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="📚 Bookstore Customer Analytics",
    layout="centered"
)

# -------------------------------
# ✅ Custom CSS Theme
# -------------------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Arial', sans-serif;
    }
    .big-title {
        font-size: 36px;
        color: #4B0082;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
    }
    .sub-header {
        font-size: 18px;
        text-align: center;
        color: #333333;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #4B0082;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        font-size: 18px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# ✅ Sidebar Navigation
# -------------------------------
st.sidebar.title("📚 Bookstore App Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📉 Customer Churn Prediction"])
st.sidebar.caption("🔍 Powered by ANN • Streamlit • SQLite")

# -------------------------------
# ✅ Load ANN Model & Scaler
# -------------------------------
@st.cache_resource
def load_churn_model():
    return load_model('model.h5')

model = load_churn_model()
scaler = joblib.load('scaler.pkl')

# -------------------------------
# ✅ DB Connection Function
# -------------------------------
def get_connection():
    return sqlite3.connect("gravity_books.db")

# -------------------------------
# ✅ HOME PAGE
# -------------------------------
if page == "\ud83c\udfe0 Home":
    st.markdown("<div class='big-title'>\ud83d\udcda BOOKSTORE CUSTOMER ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predict, analyze & retain your bookstore customers with AI \ud83d\udcc8\u2728</div>", unsafe_allow_html=True)

    st.markdown("""
    Welcome to your smart customer analytics app!  
    
    This tool uses a trained **Artificial Neural Network (ANN)** model to detect churn risk and help you grow retention.

    ---
    **\ud83d\udcca Features:**
    - :blue[Analyze customer order history]
    - :orange[Predict churn likelihood]
    - :green[Store predictions for future actions]

    ---
    **\ud83d\udca1 Why it matters:**
    Knowing which customers are likely to churn helps you:
    - Offer targeted promotions
    - Improve retention rates
    - Increase lifetime value & revenue

    \ud83d\udc49 Use the sidebar to navigate to **Customer Churn Prediction** and test it now!
    """)
    st.image("bookstore.jpeg", use_container_width=True)

# -------------------------------
# ✅ CUSTOMER CHURN PREDICTION PAGE
# -------------------------------
elif page == "\ud83d\udcc9 Customer Churn Prediction":
    st.markdown("<div class='big-title'>\ud83d\udcc9 CUSTOMER CHURN PREDICTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>\ud83d\udd0d Fetch customer data, analyze behavior & predict churn risk with AI.</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.info("\ud83d\udccc **How it works:** Enter a valid Customer ID below. The app fetches their order history, predicts churn risk, and stores it for your retention strategy.")

    def fetch_customer_data(customer_id):
        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                julianday('now') - julianday(MIN(order_date)) AS days_active,
                julianday('now') - julianday(MAX(order_date)) AS days_since_last_order,
                COUNT(DISTINCT co.order_id) AS total_orders,
                AVG(ol.price) AS avg_book_price
            FROM cust_order co
            JOIN order_line ol ON co.order_id = ol.order_id
            WHERE co.customer_id = ?;
        """
        cursor.execute(query, (customer_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row

    def predict_churn(input_data):
        input_data_scaled = scaler.transform(input_data)
        pred = model.predict(input_data_scaled)

        if pred.shape[1] == 1:
            predicted_class = int(pred[0][0] >= 0.5)
        else:
            predicted_class = int(np.argmax(pred, axis=1)[0])

        return pred, predicted_class, input_data_scaled

    with st.form("churn_form"):
        customer_id = st.number_input("\ud83d\udc64 Customer ID", min_value=1, step=1, help="Enter the numeric ID of the customer.")
        submitted = st.form_submit_button("\ud83d\udd0d Fetch & Predict")

    if submitted:
        row = fetch_customer_data(customer_id)

        if row and row[0] is not None:
            st.success("\u2705 **Customer data found!**")
            row_dict = {
                'days_active': row[0],
                'days_since_last_order': row[1],
                'total_orders': row[2],
                'avg_book_price': row[3]
            }
            st.write("\ud83d\udd0d **Fetched SQL Data:**", row_dict)

            today = datetime.date.today()

            input_data = np.array([[
                float(row_dict['avg_book_price']),
                float(row_dict['days_since_last_order']),
                float(row_dict['days_active']),
                float(row_dict['total_orders']),
                today.day,
                today.month,
                today.year
            ]])

            st.markdown("#### \ud83d\udcca **Input Data Before Scaling:**")
            st.dataframe(pd.DataFrame(input_data, columns=[
                'avg_book_price', 'days_since_last_order', 'days_active',
                'total_orders', 'day', 'month', 'year'
            ]))

            pred, predicted_class, input_data_scaled = predict_churn(input_data)

            st.markdown("#### \ud83d\udccf **Input Data After Scaling:**")
            st.dataframe(pd.DataFrame(input_data_scaled, columns=[
                'avg_book_price', 'days_since_last_order', 'days_active',
                'total_orders', 'day', 'month', 'year'
            ]))

            probability = float(pred[0][0]) if pred.shape[1] == 1 else None

            if probability is not None:
                st.info(f"\ud83d\udd2e **Churn Probability:** `{probability:.2%}`")
                st.progress(probability)
            else:
                st.info(f"\ud83d\udd2e **Raw Model Output:** `{pred.tolist()}`")

            st.write("\u2705 **Predicted Class:**", predicted_class)

            conn = get_connection()
            cursor = conn.cursor()
            save_query = """
                INSERT INTO churn_predictions (customer_id, prediction, prediction_date)
                VALUES (?, ?, ?);
            """
            cursor.execute(save_query, (customer_id, predicted_class, str(today)))
            conn.commit()
            cursor.close()
            conn.close()

            if predicted_class == 0:
                st.success("\ud83d\udfe2 **Good news!** The customer is likely to **stay**. \u2705")
                st.markdown("\ud83d\udca1 **Tip:** Consider sending loyalty rewards or thank-you notes.")
            else:
                st.warning("\ud83d\udd34 **Alert!** The customer may **churn** soon. \u26a0\ufe0f")
                st.markdown("\ud83d\udca1 **Tip:** Consider sending a discount offer or a personalized re-engagement email.")
        else:
            st.error(f"\u26a0\ufe0f No valid orders found for Customer ID `{customer_id}`. Please try another ID.")
