# Supermart-Grocery-Sales
## Project Overview

This project analyzes a fictional dataset of grocery sales from Tamil Nadu, India.  It combines data preprocessing, feature engineering, exploratory data analysis (EDA), SQL queries, and machine learning (Linear Regression ; Random Forest).  A Streamlit dashboard was developed for interactive exploration and real-time predictions.

## Dataset
- Source: Provided CSV dataset (Supermart Grocery Sales - Retail Analytics Dataset)  
- Columns: Order ID, Customer Name, Category, Sub Category, City, State, Region, Order Date, Sales, Discount, Profit  

## Tech Stack
- Languages: Python, SQL  
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib  
- Database: MySQL (via SQLAlchemy)  
- Visualization: Streamlit Dashboard, Excel exports  

## Features
- Data Preprocessing (cleaning, handling missing values, feature engineering)  
- Exploratory Data Analysis (EDA) with visualizations  
- SQL queries for deeper insights (Top cities, Category sales, Discount bands, etc.)  
- Machine Learning models:  
  - Linear Regression → R² ≈ 0.82  
  - Random Forest Regressor → R² ≈ 0.91  
- Streamlit dashboard with:  
  - EDA page → Interactive charts  
  - Prediction page → Real-time sales prediction  
  - SQL Queries page → Execute queries and view results  
  - Export page → Download processed dataset and predictions  

## Results
- Random Forest performed better than Linear Regression  
- Discount had a strong impact on Profit & Sales  
- Oil & Masala & Beverages were top-performing categories  
- Dashboard enabled interactive exploration and predictions  
