import streamlit as st #for building the web UI
import numpy as np #for numerical operations & arrays
import pandas as pd #for dataframes and data manipulation
import sqlite3 #to interact with the SQLite database
import datetime #to get today's date
import joblib #for loading a saved scaler (used in data preprocessing)
from tensorflow.keras.models import load_model #loads the saved neural network model (model.h5)

# -------------------------------
# ✅ Streamlit Page Config
# ------------------------------- Sets the web app’s title and layout style
st.set_page_config(
    page_title="📚 Bookstore Customer Analytics",
    layout="centered"
)

#Injects CSS to:
#Style fonts, titles, buttons, and sidebar
#Use custom colors like indigo for consistency and branding

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
page = st.sidebar.radio("Go to:", ["Home", "Customer Churn Prediction"]) #Navigation options (radio buttons)
st.sidebar.caption("🔍 Powered by ANN • Streamlit • SQLite") #Footer caption showing tech stack

# -------------------------------
# ✅ Load ANN Model & Scaler
# -------------------------------
@st.cache_resource #Caches the model so it loads once
def load_churn_model(): 
    return load_model('model.h5') #Neural network used for churn prediction

model = load_churn_model()
scaler = joblib.load('scaler.pkl') #Pre-trained scaler used to normalize input features

# -------------------------------
# ✅ DB Connection Function
# -------------------------------
def get_connection():
    return sqlite3.connect("bookstore_1.db") #Creates a connection to the local SQLite database file

# -------------------------------
# ✅ HOME PAGE
# -------------------------------
if page == "Home":
    st.markdown("<div class='big-title'>📚 BOOKSTORE CUSTOMER ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predict, analyze & retain your bookstore customers with AI 📈✨</div>", unsafe_allow_html=True)

    st.markdown("""
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

    st.image("bookstore.jpeg", use_container_width=True)

# -------------------------------
# ✅ CUSTOMER CHURN PREDICTION PAGE
# -------------------------------
elif page == "Customer Churn Prediction":
    st.markdown("<div class='big-title'>📉 CUSTOMER CHURN PREDICTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>🔍 Fetch customer data, analyze behavior & predict churn risk with AI.</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.info("📌 **How it works:** Enter a valid Customer ID below. The app fetches their order history, predicts churn risk, and stores it for your retention strategy.")

    def fetch_customer_data(customer_id):
        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                julianday('now') - julianday(MIN(co."order_date")) AS days_active, #how long the customer has been active
                julianday('now') - julianday(MAX(co."order_date")) AS days_since_last_order, #Measures how long it’s been since the customer last ordered
                COUNT(DISTINCT co."order_id") AS total_orders, #Counts how many distinct orders this customer has placed
                AVG(ol."price") AS avg_book_price #Calculates the average price of books the customer has bought, based on all books across all orders for the customer
                FROM cust_order co
                JOIN order_line ol ON co."order_id" = ol."order_id"
                WHERE co."customer_id" = ?
        """
        cursor.execute(query, (customer_id,))
        row = cursor.fetchone() #Retrieves the first row of the result. Since the query is an aggregation, only one row is expected
        #Closes the cursor and database connection properly
        cursor.close()
        conn.close()
        return row #Returns the result as a tuple

    def predict_churn(input_data):   #input_data is likely a NumPy array or DataFrame of shape (1, n_features) representing a single customer's features 
        input_data_scaled = scaler.transform(input_data)  #scales this data using a pre-fitted scaler (e.g., StandardScaler, MinMaxScaler)—essential for ensuring input matches the scale the model was trained on
        pred = model.predict(input_data_scaled)  #Passes the scaled input to the pre-trained model (probably a Keras ANN)

        if pred.shape[1] == 1: #If the prediction has shape (1, 1), i.e., it's binary
            predicted_class = int(pred[0][0] >= 0.5)  #Uses threshold 0.5 to decide: >= 0.5 → churned (1), < 0.5 → not churned (0)
        else:
            predicted_class = int(np.argmax(pred, axis=1)[0])  #np.argmax finds the index of the highest predicted probability

        return pred, predicted_class, input_data_scaled  #pred → Raw model prediction, predicted_class → Final class label & Scaled version of the input

    with st.form("churn_form"):
        customer_id = st.number_input("👤 Customer ID", min_value=1, step=1, help="Enter the numeric ID of the customer.")
        submitted = st.form_submit_button("🔍 Fetch & Predict")

    if submitted:
        row = fetch_customer_data(customer_id)

    #Checks if SQL query returned a valid result (row) and that the first value (days_active) is not None. Prevents further code from running if no data is found for the customer
        if row and row[0] is not None: 
            st.success("✅ **Customer data found!**")

    #Converts the SQL result row into a more readable dictionary
            row_dict = {
                'days_active': row[0],
                'days_since_last_order': row[1],
                'total_orders': row[2],
                'avg_book_price': row[3]
            }
            st.write("🔍 **Fetched SQL Data:**", row_dict)

            today = datetime.date.today()  #Gets the current date using Python’s datetime module

            input_data = np.array([[
                float(row_dict['avg_book_price']),
                float(row_dict['days_since_last_order']),
                float(row_dict['days_active']),
                float(row_dict['total_orders']),
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

            probability = float(pred[0][0]) if pred.shape[1] == 1 else None   #ANN model’s output (pred) is typically a 2D array

            if probability is not None:
                st.info(f"🔮 **Churn Probability:** `{probability:.2%}`")  #It’s displayed in percentage form using :.2% 
                st.progress(probability)  #st.progress() visualizes it with a horizontal progress bar in the UI
            else:
                st.info(f"🔮 **Raw Model Output:** `{pred.tolist()}`")   #That means the model isn't a typical binary classifier. So instead, it just prints the raw model output

            st.write("✅ **Predicted Class:**", predicted_class)

            conn = get_connection() #Calls your previously defined get_connection() function. This creates a connection to the bookstore_1.db SQLite database
            cursor = conn.cursor()  #Creates a cursor object. The cursor lets you execute SQL commands on the database
            save_query = """
                INSERT INTO churn_predictions (customer_id, prediction, prediction_date)
                VALUES (?, ?, ?); 
            """  #Prepares an SQL INSERT statement using parameter placeholders (?) to avoid SQL injection
            cursor.execute(save_query, (customer_id, predicted_class, str(today)))   #Executes the prepared query with actual values. str(today) converts the datetime.date object to a string
            conn.commit()  #Commits the transaction to the database. Without this line, the insert would not be saved permanently
            cursor.close()
            conn.close() #Closes the cursor and connection to release resources and avoid DB locks

            if predicted_class == 0:  #Checks if the predicted class is 0, which means the customer is not likely to churn
                st.success("🟢 **Good news!** The customer is likely to **stay**. ✅")
                st.markdown("💡 **Tip:** Consider sending loyalty rewards or thank-you notes.")
            else:
                st.warning("🔴 **Alert!** The customer may **churn** soon. ⚠️")
                st.markdown("💡 **Tip:** Consider sending a discount offer or a personalized re-engagement email.")
        else:
            st.error(f"⚠️ No valid orders found for Customer ID `{customer_id}`. Please try another ID.")
