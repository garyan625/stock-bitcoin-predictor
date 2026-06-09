const response = await axios.post('https://yelling-supplier-open.ngrok-free.dev/predict', {
  history: [ /* your last 60 days of closing prices */ ]
});
console.log("Prediction for tomorrow:", response.data.prediction);