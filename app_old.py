import streamlit as st # For creating interactive Streamlit apps
import numpy as np # For numerical operations & arrays
import pandas as pd # For dataframes and data manipulation
import mysql.connector # For connecting to MySQL database
import datetime # To get today’s date
import joblib # For loading the pre-fitted scaler
from tensorflow.keras.models import load_model # To load your ANN model

# -------------------------------
# ✅ Streamlit Page Config  Sets your Streamlit page’s title in the browser tab and centers the layout for a neat look
# -------------------------------
st.set_page_config(
    page_title="📚 Bookstore Customer Analytics",
    layout="centered"
)

# -------------------------------
# ✅ Custom CSS Theme  Adds custom HTML/CSS to style your headings, buttons, status boxes, and sidebar
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
page = st.sidebar.radio("Go to:", ["🏠 Home", "📉 Customer Churn Prediction"])    #Adds a sidebar title and a radio button for page navigation
st.sidebar.caption("🔍 Powered by ANN • Streamlit • MySQL")

# -------------------------------
# ✅ Load ANN Model & Scaler  Loads your scaler.pkl & model.h5 which ensures your input data matches how the model was trained
# -------------------------------
@st.cache_resource     #Uses Streamlit’s @st.cache_resource to load your ANN model only once (cached)
def load_churn_model():
    return load_model('model.h5')

model = load_churn_model()
scaler = joblib.load('scaler.pkl')

# -------------------------------
# ✅ DB Connection Function
# -------------------------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="GovaMoni!Kanali26",
        database="gravity_books"
    )

# -------------------------------
# ✅ HOME PAGE
# -------------------------------
if page == "🏠 Home":
    st.markdown("<div class='big-title'>📚 BOOKSTORE CUSTOMER ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predict, analyze & retain your bookstore customers with AI 📈✨</div>", unsafe_allow_html=True)

    st.markdown("""
    Welcome to your smart customer analytics app!  
    
    This tool uses a trained **Artificial Neural Network (ANN)** model to detect churn risk and help you grow retention.

    ---
    **📊 Features:**
    - :blue[Analyze customer order history]
    - :orange[Predict churn likelihood]
    - :green[Store predictions for future actions]

    ---
    **💡 Why it matters:**
    Knowing which customers are likely to churn helps you:
    - Offer targeted promotions
    - Improve retention rates
    - Increase lifetime value & revenue

    👉 Use the sidebar to navigate to **Customer Churn Prediction** and test it now!
    """)
    st.image("D:/Gova/Data Science Course/Projects/4. Neural Networks for the Publishing Industry/bookstore.jpeg", use_container_width=True)

# -------------------------------
# ✅ CUSTOMER CHURN PREDICTION PAGE (Enhanced)
# -------------------------------
elif page == "📉 Customer Churn Prediction":
    st.markdown("<div class='big-title'>📉 CUSTOMER CHURN PREDICTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>🔍 Fetch customer data, analyze behavior & predict churn risk with AI.</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.info("📌 **How it works:** Enter a valid Customer ID below. The app fetches their order history, predicts churn risk, and stores it for your retention strategy.")

    # ✅ Helper: Fetch Customer Data
    def fetch_customer_data(customer_id):
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT
                DATEDIFF(CURDATE(), MIN(order_date)) AS days_active,
                DATEDIFF(CURDATE(), MAX(order_date)) AS days_since_last_order,
                COUNT(DISTINCT co.order_id) AS total_orders,
                AVG(ol.price) AS avg_book_price
            FROM cust_order co
            JOIN order_line ol ON co.order_id = ol.order_id
            WHERE co.customer_id = %s;
        """
        cursor.execute(query, (customer_id,))
        row = cursor.fetchone()

        cursor.close()
        conn.close()
        return row   #Returns the data as a dictionary

    # ✅ Helper: Predict Churn
    def predict_churn(input_data):
        input_data_scaled = scaler.transform(input_data)  #Scales your input data
        pred = model.predict(input_data_scaled)  #Predicts churn using your ANN model

        if pred.shape[1] == 1:
            predicted_class = int(pred[0][0] >= 0.5)
        else:
            predicted_class = int(np.argmax(pred, axis=1)[0])

        return pred, predicted_class, input_data_scaled

    # ✅ Input Form
    with st.form("churn_form"):
        customer_id = st.number_input("👤 Customer ID", min_value=1, step=1, help="Enter the numeric ID of the customer.")
        submitted = st.form_submit_button("🔍 Fetch & Predict")

    if submitted:
        row = fetch_customer_data(customer_id)

        if row and row['avg_book_price'] is not None:
            st.success("✅ **Customer data found!**")
            st.write("🔍 **Fetched SQL Data:**", row)

            today = datetime.date.today()

            input_data = np.array([[
                float(row['avg_book_price']),
                float(row['days_since_last_order']),
                float(row['days_active']),
                float(row['total_orders']),
                today.day,
                today.month,
                today.year
            ]])

            st.markdown("#### 📊 **Input Data Before Scaling:**")
            st.dataframe(pd.DataFrame(input_data, columns=[
                'avg_book_price', 'days_since_last_order', 'days_active',
                'total_orders', 'day', 'month', 'year'
            ]))

            pred, predicted_class, input_data_scaled = predict_churn(input_data)

            st.markdown("#### 📏 **Input Data After Scaling:**")
            st.dataframe(pd.DataFrame(input_data_scaled, columns=[
                'avg_book_price', 'days_since_last_order', 'days_active',
                'total_orders', 'day', 'month', 'year'
            ]))

            # Nicely formatted output
            probability = float(pred[0][0]) if pred.shape[1] == 1 else None

            if probability is not None:
                st.info(f"🔮 **Churn Probability:** `{probability:.2%}`")
                st.progress(probability)
            else:
                st.info(f"🔮 **Raw Model Output:** `{pred.tolist()}`")

            st.write("✅ **Predicted Class:**", predicted_class)

            # ✅ Save to DB
            conn = get_connection()
            cursor = conn.cursor()
            save_query = """
                INSERT INTO churn_predictions (customer_id, prediction, prediction_date)
                VALUES (%s, %s, NOW());
            """
            cursor.execute(save_query, (customer_id, predicted_class))
            conn.commit()
            cursor.close()
            conn.close()

            # ✅ Final Result
            if predicted_class == 0:
                st.success("🟢 **Good news!** The customer is likely to **stay**. ✅")
                st.markdown("💡 **Tip:** Consider sending loyalty rewards or thank-you notes.")
            else:
                st.warning("🔴 **Alert!** The customer may **churn** soon. ⚠️")
                st.markdown("💡 **Tip:** Consider sending a discount offer or a personalized re-engagement email.")

        else:
            st.error(f"⚠️ No valid orders found for Customer ID `{customer_id}`. Please try another ID.")
