from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os

# Define the correct path to your model
print(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(__file__), 'src/models', 'iris_model.pkl')
print(model_path)

# Load the model
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)


# Define the FastAPI app
app = FastAPI()

# Define a data model for the input
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define an endpoint for model predictions
@app.post("/predict")
def predict(data: IrisData):
    # Convert input data to the format the model expects
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
