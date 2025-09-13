# Import Required Libraries
import os, re
import io
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from pathlib import Path

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Config / Paths
DATA_PATH = r"C:/Users/Hp/OneDrive/Desktop/python/supermartgrocerysales/Supermart Grocery Sales - Retail Analytics Dataset.csv"
data = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/supermartgrocerysales/Supermart Grocery Sales - Retail Analytics Dataset.csv")
print("Data loaded successfully!")
print(data.head())

LIN_MODEL_FP = "lin_model.pkl"
RF_MODEL_FP = "rf_model.pkl"
SCALER_FP = "scaler.pkl"
ENCODERS_FP = "label_encoders.pkl"

# MySQL connection 
MYSQL_CONN = "mysql+pymysql://root:Abhi%40100982@localhost/stock_market"

TABLE_NAME = "supermart_sales"
EXCEL_PATH = Path("C:/Users/Hp/OneDrive/Desktop/python/supermartgrocerysales/supermart_processed.xlsx")

# Helper: Excel export
def convert_df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return out.getvalue()

# -----------------------
# Load & preprocess (shared)
# -----------------------
def load_and_preprocess(path=DATA_PATH, save_engine=None):
    df = pd.read_csv(path)
    # Basic cleaning
    df = df.copy()
    # drop blank rows explicitly
    df.dropna(how="all", inplace=True)

    # Parse dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)

    # Remove exact duplicates if any
    df.drop_duplicates(inplace=True)

    # Feature engineering
    df['month_no'] = df['Order Date'].dt.month
    df['Day_Of_Week'] = df['Order Date'].dt.dayofweek
    df['Profit_Margin'] = np.where(df['Sales'] != 0, df['Profit'] / df['Sales'], 0)
    df['Has_Discount'] = np.where(df['Discount'] > 0, 1, 0)
    df['Discount_Profit_Interaction'] = df['Discount'] * df['Profit']
    df['Order Day'] = df['Order Date'].dt.day
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.strftime('%B')

    # Fill or drop any remaining NaNs if present (safer to drop here)
    df.dropna(inplace=True)

    # Encode categorical columns and keep label encoders
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Optionally save cleaned df to MySQL
    if save_engine is not None:
        try:
            df.to_sql("supermart_sales", con=save_engine, if_exists="replace", index=False)
            print("Saved processed dataset to MySQL table 'supermart_sales'.")
        except Exception as e:
            print("Could not save to MySQL:", e)

    return df, label_encoders

# -----------------------
# Train models (if needed)
# -----------------------
def train_and_save_models(df, label_encoders, force_retrain=False):
    # define features used for modeling (must match what the app will send)
    feature_cols = [
        'Category', 'Sub Category', 'City', 'Region', 'State',
        'Discount', 'Profit', 'month_no', 'Profit_Margin',
        'Has_Discount', 'Day_Of_Week', 'Discount_Profit_Interaction'
    ]

    # Ensure columns available
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for training: {missing}")

    X = df[feature_cols]
    y = df['Sales']

    # Train only if artifacts are missing or force_retrain True
    need_train = force_retrain or not (os.path.exists(RF_MODEL_FP) and os.path.exists(LIN_MODEL_FP) and os.path.exists(SCALER_FP) and os.path.exists(ENCODERS_FP))

    if need_train:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        # Fit scaler using DataFrame (preserves feature_names_in_)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Linear Regression
        lin = LinearRegression()
        lin.fit(X_train_scaled, y_train)
        lin_pred = lin.predict(X_test_scaled)
        lin_mse = mean_squared_error(y_test, lin_pred)
        lin_r2 = r2_score(y_test, lin_pred)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)

        # Save artifacts
        joblib.dump(lin, LIN_MODEL_FP)
        joblib.dump(rf, RF_MODEL_FP)
        joblib.dump(scaler, SCALER_FP)
        joblib.dump(label_encoders, ENCODERS_FP)

        print("Models trained & saved.")
        metrics = {
            "lin_mse": lin_mse, "lin_r2": lin_r2,
            "rf_mse": rf_mse, "rf_r2": rf_r2
        }
        return metrics
    else:
        # If already saved, load models and compute metrics on full data (or skip)
        scaler = joblib.load(SCALER_FP)
        lin = joblib.load(LIN_MODEL_FP)
        rf = joblib.load(RF_MODEL_FP)
        X_scaled = scaler.transform(X)
        lin_pred = lin.predict(X_scaled)
        rf_pred = rf.predict(X_scaled)
        lin_r2 = r2_score(y, lin_pred)
        lin_mse = mean_squared_error(y, lin_pred)
        rf_r2 = r2_score(y, rf_pred)
        rf_mse = mean_squared_error(y, rf_pred)
        print("Loaded pre-trained models.")
        return {"lin_mse": lin_mse, "lin_r2": lin_r2, "rf_mse": rf_mse, "rf_r2": rf_r2}


# -----------------------
# MAIN: Prepare data, models, save to MySQL (if available)
# -----------------------
def prepare_everything():
    # optional: attempt to create engine (skip MySQL if not available)
    engine = None
    try:
        engine = create_engine(MYSQL_CONN)
    except Exception as e:
        print("Warning: Could not create MySQL engine. MySQL features will be skipped.", e)

    df, label_encoders = load_and_preprocess(DATA_PATH, save_engine=engine)
    metrics = train_and_save_models(df, label_encoders, force_retrain=False)

    # If engine exists, ensure predictions table exists (create empty if not)
    if engine is not None:
        try:
            # create predictions table if not exists with appropriate columns
            sample_pred_df = pd.DataFrame([{
                "Category": None, "Sub_Category": None, "City": None, "Region": None, "State": None,
                "Discount": None, "Profit": None, "Month_No": None, "Predicted_Sales": None, "Timestamp": None
            }])
            sample_pred_df.to_sql("supermart_predictions", con=engine, if_exists="append", index=False)
            # Remove the sample row (if any) to keep table clean
            with engine.begin() as conn:
                conn.execute("DELETE FROM supermart_predictions WHERE Predicted_Sales IS NULL")
        except Exception:
            pass

    return df, label_encoders, metrics, engine

# Prepare once when running as script (and before Streamlit UI loads)
df_processed, label_encoders, model_metrics, engine = prepare_everything()

# SQLAlchemy -> MySQL

engine = create_engine(MYSQL_CONN, echo=False, future=True)

Total_Sales_by_Category = text("""
SELECT Category, SUM(Sales) AS Total_Sales
FROM supermart_sales
GROUP BY Category
ORDER BY Total_Sales DESC;
""")
Top_5_Cities_by_Sales = text("""
SELECT City, SUM(Sales) AS Total_Sales
FROM supermart_sales
GROUP BY City
ORDER BY Total_Sales DESC
LIMIT 5;
""")
Monthly_Sales_Trend = text("""
SELECT DATE_FORMAT(`Order Date`, '%Y-%m') AS Month,
    SUM(Sales) AS Total_Sales
FROM supermart_sales
GROUP BY DATE_FORMAT(`Order Date`, '%Y-%m')
ORDER BY Month;
""")
Profit_by_Discount_Range = text("""
SELECT 
    CASE 
        WHEN Discount = 0 THEN 'No Discount'
        WHEN Discount BETWEEN 0.01 AND 0.20 THEN 'Low Discount'
        WHEN Discount BETWEEN 0.21 AND 0.50 THEN 'Medium Discount'
        ELSE 'High Discount'
    END AS Discount_Band,
    SUM(Profit) AS Total_Profit
FROM supermart_sales
GROUP BY Discount_Band
ORDER BY Total_Profit DESC;
""")
Customer_Segment_Performance = text("""
SELECT Region AS Segment,
       SUM(Sales) AS Total_Sales,
       SUM(Profit) AS Total_Profit
FROM supermart_sales
GROUP BY Region
ORDER BY Total_Sales DESC;
""")
with engine.connect() as conn:
        Total_Sales_by_Category = pd.read_sql(Total_Sales_by_Category, conn)
        Top_5_Cities_by_Sales = pd.read_sql(Top_5_Cities_by_Sales, conn)
        Monthly_Sales_Trend = pd.read_sql(Monthly_Sales_Trend, conn)
        Profit_by_Discount_Range = pd.read_sql(Profit_by_Discount_Range, conn)
        Customer_Segment_Performance = pd.read_sql(Customer_Segment_Performance, conn)
print("\nSQL Results:")
print(Total_Sales_by_Category.head())
print(Top_5_Cities_by_Sales.head())
print(Monthly_Sales_Trend.head())
print(Profit_by_Discount_Range.head())
print(Customer_Segment_Performance.head())

# -----------------------
# STREAMLIT APP
# -----------------------
# App layout
st.set_page_config(page_title="Supermart Sales Dashboard", layout="wide")
st.title("🛒 Supermart Grocery Sales — Dashboard & Sales Prediction")

# Cache loading of models/encoders/scaler
@st.cache_resource
def load_artifacts():
    lin = joblib.load(LIN_MODEL_FP)
    rf = joblib.load(RF_MODEL_FP)
    scaler = joblib.load(SCALER_FP)
    encs = joblib.load(ENCODERS_FP)
    return lin, rf, scaler, encs

lin_model, rf_model, scaler, loaded_label_encoders = load_artifacts()


# Sidebar for navigation & options
st.sidebar.header("Navigation")
page = st.sidebar.radio("", ["EDA", "Prediction", "Data Export", "DB & Info", "SQL Queries"])

# -----------------------
# EDA PAGE
# -----------------------
if page == "EDA":
    st.header("Exploratory Data Analysis")

    # Show small data preview
    st.subheader("Data preview")
    st.dataframe(df_processed.head(100))

    # Sales by Category (bar)
    st.subheader("Sales by Category")
    fig1, ax1 = plt.subplots(figsize=(8,4))
    # decode label numbers to labels for plotting (if encoders present)
    if 'Category' in loaded_label_encoders:
        cat_labels = loaded_label_encoders['Category'].inverse_transform(df_processed['Category'])
        plot_df = pd.DataFrame({'Category': cat_labels, 'Sales': df_processed['Sales']})
        sns.barplot(data=plot_df.groupby('Category', as_index=False).sum().sort_values('Sales', ascending=False), x='Category', y='Sales', ax=ax1)
        ax1.tick_params(axis='x', rotation=45)
    else:
        sns.barplot(x='Category', y='Sales', data=df_processed, estimator=sum, ax=ax1)
    st.pyplot(fig1)

    # Sales Trend Over Time
    st.subheader("Sales Trend Over Time")
    sales_trend = df_processed.groupby('Order Date', as_index=False)['Sales'].sum().sort_values('Order Date')
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(sales_trend['Order Date'], sales_trend['Sales'], marker='o', linestyle='-')
    ax2.set_xlabel("Order Date")
    ax2.set_ylabel("Sales")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Correlation heatmap (using numeric columns)
    st.subheader("Correlation Heatmap")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    corr = df_processed[numeric_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Top cities
    st.subheader("Top 5 Cities by Sales")
    # decode city names if encoder present
    if 'City' in loaded_label_encoders:
        city_labels = loaded_label_encoders['City'].inverse_transform(df_processed['City'])
        tmp = pd.DataFrame({'City': city_labels, 'Sales': df_processed['Sales']}).groupby('City').sum().sort_values('Sales', ascending=False).head(5)
        fig4, ax4 = plt.subplots(figsize=(8,4))
        tmp['Sales'].plot(kind='bar', ax=ax4)
        ax4.set_ylabel("Sales")
        ax4.set_xlabel("City")
        plt.xticks(rotation=45)
        st.pyplot(fig4)
    else:
        top_cities = df_processed.groupby('City')['Sales'].sum().sort_values(ascending=False).head(5)
        fig4, ax4 = plt.subplots(figsize=(8,4))
        top_cities.plot(kind='bar', ax=ax4)
        st.pyplot(fig4)

# -----------------------
# PREDICTION PAGE
# -----------------------
elif page == "Prediction":
    st.header("Real-time Sales Prediction")

    # Choose model
    model_choice = st.selectbox("Choose model", ["Random Forest", "Linear Regression"])
    model = rf_model if model_choice == "Random Forest" else lin_model

    # Show model metrics
    st.subheader("Model performance (on processed data)")
    st.write(f"Linear Regression — R²: {model_metrics['lin_r2']:.4f}, MSE: {model_metrics['lin_mse']:.2f}")
    st.write(f"Random Forest — R²: {model_metrics['rf_r2']:.4f}, MSE: {model_metrics['rf_mse']:.2f}")

    # Input widgets (use the original class labels for dropdowns)
    def get_classes(col):
        if col in loaded_label_encoders:
            return list(loaded_label_encoders[col].classes_)
        else:
            return []

    category = st.selectbox("Category", get_classes('Category'))
    sub_category = st.selectbox("Sub Category", get_classes('Sub Category'))
    city = st.selectbox("City", get_classes('City'))
    region = st.selectbox("Region", get_classes('Region'))
    state = st.selectbox("State", get_classes('State'))
    discount = st.number_input("Discount (0-1)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
    profit = st.number_input("Profit", value=0.0, step=0.01)
    month_no = st.number_input("Month number (1-12)", min_value=1, max_value=12, value=1)

    # Feature engineering for single input
    profit_margin = profit / 1 if profit != 0 else 0
    has_discount = 1 if discount > 0 else 0
    day_of_week = 0  # optional: could add input for day of week
    discount_profit_interaction = discount * profit

    # Encode categorical selections using saved encoders
    try:
        enc_cat = loaded_label_encoders['Category'].transform([category])[0]
        enc_sub = loaded_label_encoders['Sub Category'].transform([sub_category])[0]
        enc_city = loaded_label_encoders['City'].transform([city])[0]
        enc_region = loaded_label_encoders['Region'].transform([region])[0]
        enc_state = loaded_label_encoders['State'].transform([state])[0]
    except Exception as e:
        st.error("Error encoding categorical input. Make sure dropdown values are present in label encoders.")
        st.stop()

    input_df = pd.DataFrame([[
        enc_cat, enc_sub, enc_city, enc_region, enc_state,
        discount, profit, month_no, profit_margin, has_discount, day_of_week, discount_profit_interaction
    ]], columns=[
        'Category', 'Sub Category', 'City', 'Region', 'State',
        'Discount', 'Profit', 'month_no', 'Profit_Margin',
        'Has_Discount', 'Day_Of_Week', 'Discount_Profit_Interaction'
    ])

    # Scale and predict
    try:
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        st.subheader("Predicted Sales")
        st.success(f"₹{pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Small bar chart for predicted value
    figp, axp = plt.subplots(figsize=(4,3))
    axp.bar(["Predicted Sales"], [pred])
    axp.set_ylim(0, max(pred * 1.2, 100))
    st.pyplot(figp)

    # Create a DataFrame for this prediction (decoded human-readable)
    pred_log = pd.DataFrame([{
        "Category": category,
        "Sub_Category": sub_category,
        "City": city,
        "Region": region,
        "State": state,
        "Discount": discount,
        "Profit": profit,
        "Month_No": month_no,
        "Predicted_Sales": pred,
        "Timestamp": datetime.datetime.now()
    }])

    # Log to MySQL (if engine available)
    if engine is not None:
        try:
            pred_log.to_sql("supermart_predictions", con=engine, if_exists="append", index=False)
            st.info("Prediction logged in MySQL (table: supermart_predictions).")
        except Exception as e:
            st.warning(f"Could not log prediction to MySQL: {e}")

    # Download single prediction as Excel
    excel_bytes = convert_df_to_excel_bytes(pred_log)
    st.download_button("Download prediction (Excel)", data=excel_bytes, file_name="prediction_record.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------
# DATA EXPORT PAGE
# -----------------------
elif page == "Data Export":
    st.header("Exports")

    # Download processed dataset as Excel
    st.subheader("Download processed dataset (Excel)")
    processed_bytes = convert_df_to_excel_bytes(df_processed)
    st.download_button("Download processed dataset", data=processed_bytes, file_name="supermart_processed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Download predictions from DB (if engine available)
    if engine is not None:
        st.subheader("Download predictions table from MySQL")
        try:
            pred_df = pd.read_sql("SELECT * FROM supermart_predictions", con=engine)
            st.write(f"Found {len(pred_df)} prediction rows")
            st.dataframe(pred_df.tail(50))
            bytes_all = convert_df_to_excel_bytes(pred_df)
            st.download_button("Download predictions (Excel)", data=bytes_all, file_name="supermart_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.warning(f"Could not read predictions from MySQL: {e}")

# -----------------------
# DB & Info PAGE
# -----------------------
elif page == "DB & Info":
    st.header("Database & Model Info")
    st.markdown("**Model metrics (estimated on processed data):**")
    st.write(f"Linear Regression R²: {model_metrics['lin_r2']:.4f}, MSE: {model_metrics['lin_mse']:.2f}")
    st.write(f"Random Forest R²: {model_metrics['rf_r2']:.4f}, MSE: {model_metrics['rf_mse']:.2f}")

    st.markdown("**Saved artifacts**")
    st.write(f"- Linear model: {LIN_MODEL_FP}")
    st.write(f"- Random Forest model: {RF_MODEL_FP}")
    st.write(f"- Scaler: {SCALER_FP}")
    st.write(f"- Label encoders: {ENCODERS_FP}")

    st.markdown("**MySQL connection**")
    st.write(f"- Connection string used: {MYSQL_CONN}")
    if engine is not None:
        st.success("MySQL engine available")
    else:
        st.error("MySQL engine not available (check MYSQL_CONN & dependencies)")

# SQL Queries Section

elif page == "SQL Queries":
    st.header("Run SQL Queries")

    query_type = st.selectbox("Select Query", [
        "Total_Sales_by_Category",
        "Top_5_Cities_by_Sales",
        "Monthly_Sales_Trend",
        "Profit_by_Discount_Range",
        "Customer_Segment_Performance"
    ])

    engine = create_engine(MYSQL_CONN, echo=False, future=True)

    if query_type == "Total_Sales_by_Category":
        sql = text("""
            SELECT Category, SUM(Sales) AS Total_Sales
            FROM supermart_sales
            GROUP BY Category
            ORDER BY Total_Sales DESC;
        """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn)
        st.write(results)

    elif query_type == "Top_5_Cities_by_Sales":
        sql = text("""
            SELECT City, SUM(Sales) AS Total_Sales
            FROM supermart_sales
            GROUP BY City
            ORDER BY Total_Sales DESC
            LIMIT 5;
            """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn)
        st.write(results)

    elif query_type == "Monthly_Sales_Trend":
        sql = text("""
            SELECT DATE_FORMAT(`Order Date`, '%Y-%m') AS Month,
            SUM(Sales) AS Total_Sales
            FROM supermart_sales
            GROUP BY DATE_FORMAT(`Order Date`, '%Y-%m')
            ORDER BY Month;
        """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn)
        st.write(results)
            
    elif query_type == "Profit_by_Discount_Range":
        sql = text("""
            SELECT 
                CASE 
                    WHEN Discount = 0 THEN 'No Discount'
                    WHEN Discount BETWEEN 0.01 AND 0.20 THEN 'Low Discount'
                    WHEN Discount BETWEEN 0.21 AND 0.50 THEN 'Medium Discount'
                    ELSE 'High Discount'
                END AS Discount_Band,
                SUM(Profit) AS Total_Profit
            FROM supermart_sales
            GROUP BY Discount_Band
            ORDER BY Total_Profit DESC;
        """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn)
        st.write(results)
        
    elif query_type == "Customer_Segment_Performance":
        sql = text("""
            SELECT Region AS Segment,
                   SUM(Sales) AS Total_Sales,
                   SUM(Profit) AS Total_Profit
            FROM supermart_sales
            GROUP BY Region
            ORDER BY Total_Sales DESC;
            """)
        with engine.connect() as conn:
            results = pd.read_sql(sql, conn)
        st.write(results)



# End of app
