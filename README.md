# Crypto-Stock Predictor: Cloud-Deployed AI Pipeline

## Project Overview
A sophisticated, full-stack predictive analytics platform designed to forecast BTC-USD price trends. This project transitions from a local data science model to a scalable, cloud-deployed inference engine, demonstrating proficiency in both AI/ML workflows and modern DevOps practices.

## Key Features
* **Predictive Analytics:** Implemented a Deep Learning model using an **LSTM (Long Short-Term Memory)** architecture in PyTorch to analyze time-series financial data.
* **Cloud-Native Deployment:** Overcame local hardware constraints by architecting a cloud-hosted inference engine on **Google Colab**.
* **RESTful API Integration:** Deployed a high-performance **FastAPI** backend to serve model predictions in real-time.
* **Secure Tunneling:** Leveraged **ngrok** to establish a secure bridge between cloud-hosted inference and a local React frontend.
* **Performance Metrics:** Achieved a **MAPE (Mean Absolute Percentage Error) of 11.97%**, demonstrating high predictive reliability for volatile asset forecasting.

## Tech Stack
* **AI/ML:** PyTorch, LSTM, Scikit-Learn, NumPy, Pandas
* **Data Source:** `yfinance` API
* **Backend:** FastAPI, Uvicorn
* **Deployment/Tunneling:** Google Colab, ngrok
* **Frontend:** React.js, Axios

## Architecture Summary
The system follows a decoupled architecture where the training and inference logic resides in a cloud environment (Google Colab), exposed via a FastAPI REST endpoint, and integrated into a React.js dashboard via asynchronous API calls.

## How to Run (Deployment Guide)
1. **Model Training:** Execute the training notebook in Google Colab to generate `bitcoin_model.pth`.
2. **Backend API:** Host the model inference code using the provided FastAPI snippet in the Colab notebook.
3. **Tunneling:** Initialize the `ngrok` tunnel to generate a public URL.
4. **Frontend Integration:** Update the API endpoint in your React project with the generated public URL (e.g., `https://<random-string>.ngrok-free.dev/predict`).

## Professional Experience Gained
* **System Design:** Transitioned a model from local development to a cloud-based inference architecture.
* **Environment Management:** Managed complex dependency handling and asynchronous event loops (`nest-asyncio`).
* **Full-Stack Deployment:** Integrated Python-based ML models with JavaScript-based web interfaces.
