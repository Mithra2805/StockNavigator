import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Streamlit app title and instructions
st.title("Stock Price Prediction App")
st.write("This app predicts stock price movement (up/down) based on historical stock data.")

# Upload stock data (CSV file)
url = r"C:\Users\admin\Documents\Projects\Stock Prediction Project\Stock prediction dataset.csv"
if url is not None:
    # Load the dataset
    fd = pd.read_csv(url)
    
    # Display the first few rows of the dataset
    #st.write("First 5 rows of the dataset:", fd.head())
    
    # Check if the required columns are available
    if not all(col in fd.columns for col in ['Open', 'Close', 'High', 'Low', 'Date']):
        st.error("CSV must contain 'Open', 'Close', 'High', 'Low', and 'Date' columns.")
    else:
        # Preprocessing and feature extraction
        def normalize_date(date):
            try:
                return pd.to_datetime(date, errors='coerce')
            except ValueError:
                return pd.to_datetime(date, errors='coerce')

        fd['Date'] = fd['Date'].apply(normalize_date)

        # Split the 'Date' column into 'Month' and 'Year'
        fd['Month'] = fd['Date'].dt.month
        fd['Year'] = fd['Date'].dt.year

        # Create the 'target' column (stock price movement)
        fd['target'] = np.where(fd['Close'].shift(-1) > fd['Close'], 1, 0)

        # Create new features: open-close difference, low-high difference, and quarter-end flag
        fd['open-close'] = fd['Open'] - fd['Close']
        fd['low-high'] = fd['Low'] - fd['High']
        fd['is_quarter_end'] = np.where(fd['Date'].dt.month.isin([3, 6, 9, 12]) & (fd['Date'].dt.day == 31), 1, 0)

        # Select features and target
        features = fd[['open-close', 'low-high', 'is_quarter_end']]
        target = fd['target']

        # Standardize the features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Train-test split
        X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Classifier": SVC(kernel='poly', probability=True),
            "XGBoost": XGBClassifier()
        }

        # Create a section for user to choose which model to use
        model_choice = st.selectbox("Select a model", list(models.keys()))

        # Get the selected model
        model = models[model_choice]

        # Train the selected model
        model.fit(X_train, Y_train)

        # Get the ROC AUC scores
        train_auc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
        valid_auc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1])

        # Display results
        #st.write(f"Training AUC: {train_auc:.4f}")
        #st.write(f"Validation AUC: {valid_auc:.4f}")

        # Plot confusion matrix for the selected model
        y_pred = model.predict(X_valid)
        cm = metrics.confusion_matrix(Y_valid, y_pred)
        fig, ax = plt.subplots()
        sb.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #st.pyplot(fig)

        # Prediction for a custom user input (optional)
        st.header("Predict Stock Movement for Custom Data")

        # User input form for custom stock data (for simplicity, we take just open, close, high, low)
        open_price = st.number_input("Open Price", min_value=0.0, value=100.0, step=1.0)
        close_price = st.number_input("Close Price", min_value=0.0, value=100.0, step=1.0)
        high_price = st.number_input("High Price", min_value=0.0, value=100.0, step=1.0)
        low_price = st.number_input("Low Price", min_value=0.0, value=100.0, step=1.0)

        # Option for custom quarter-end flag
        is_quarter_end = st.checkbox("Is Quarter-End?", value=False)

        # Create feature vector for custom data
        custom_features = np.array([[open_price - close_price, low_price - high_price, 1 if is_quarter_end else 0]])  
        custom_features_scaled = scaler.transform(custom_features)

        # Add "Predict" button
        if st.button('Predict'):
            # Make prediction
            prediction = model.predict(custom_features_scaled)
            prediction_prob = model.predict_proba(custom_features_scaled)[:, 1]

            if prediction == 1:
                st.write("The predicted stock movement is: **Up**")
            else:
                st.write("The predicted stock movement is: **Down**")
            
            st.write(f"Prediction probability (Up): {prediction_prob[0]:.4f}")
