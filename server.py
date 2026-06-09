from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np

# 1. Define the model class (Must match the architecture used in training)
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

# 2. Initialize FastAPI
app = FastAPI()

# 3. Enable CORS (Allows React to connect to FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Load the model
model = LSTMModel()
model.load_state_dict(torch.load("bitcoin_model.pth"))
model.eval()

# 5. API Route
@app.post("/predict")
async def predict(data: dict):
    try:
        # Receive 60 days of prices from React
        # Expected input format: {"history": [val1, val2, ... val60]}
        prices = np.array(data['history']).reshape(1, 60, 1)
        input_tensor = torch.tensor(prices, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(input_tensor)
        
        return {"prediction": float(prediction.item())}
    except Exception as e:
        return {"error": str(e)}

# To run this: uvicorn server:app --reload