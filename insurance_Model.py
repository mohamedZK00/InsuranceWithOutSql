  
import os
import pandas as pd
import joblib
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# إنشاء التطبيق
app = FastAPI()

model_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(model_dir, '1-Insurance_model')

print(f"Loading model from: {model_path}")

try:
    model = load_model(model_path)
except Exception as e:
    print(f"Failed to load model with pycaret: {e}")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully with joblib.")
    except Exception as e:
        print(f"Failed to load model with joblib: {e}")
        raise e

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد نطاقات معينة هنا
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تعريف نماذج Pydantic للإدخال والإخراج
class InputModel(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class OutputModel(BaseModel):
    prediction: float

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Insurance Cost Prediction</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                background-image: url('https://img.freepik.com/free-vector/medical-healthcare-shield-background-virus-germs-protection_1017-24481.jpg?t=st=1728432737~exp=1728436337~hmac=f7a3db5738c0674fe99a515d4b5eb950ef75dfdb8bc0405e40e53a86299abd06&w=1380');
                background-size: cover;
                background-position: center;
                color: #fff;
            }
            .card {
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                background-color: rgba(255, 255, 255, 0.9);
                width: 33%;
                margin: auto;
            }
            h1 {
                color: #ffffff;
                font-size: 2.5rem;
            }
            h3 {
                color: #28a745;
            }
            label {
                color: #333;
            }
            .form-control {
                width: 100%;
                height: 38px;
                font-size: 0.9rem;
                margin: auto;
            }
            .btn-primary {
                background-color: #007bff;
                border-color: #007bff;
            }
            .btn-primary:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mt-5">Insurance Cost Prediction</h1>
            <div class="card mt-4">
                <div class="card-body">
                    <form id="predictionForm">
                        <div class="form-group">
                            <label for="age">Age:</label>
                            <input type="number" class="form-control" id="age" required>
                        </div>
                        <div class="form-group">
                            <label for="sex">Sex:</label>
                            <select class="form-control" id="sex" required>
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="bmi">BMI:</label>
                            <input type="number" step="0.1" class="form-control" id="bmi" required>
                        </div>
                        <div class="form-group">
                            <label for="children">Children:</label>
                            <input type="number" class="form-control" id="children" required>
                        </div>
                        <div class="form-group">
                            <label for="smoker">Smoker:</label>
                            <select class="form-control" id="smoker" required>
                                <option value="no">No</option>
                                <option value="yes">Yes</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="region">Region:</label>
                            <select class="form-control" id="region" required>
                                <option value="northeast">Northeast</option>
                                <option value="northwest">Northwest</option>
                                <option value="southeast">Southeast</option>
                                <option value="southwest">Southwest</option>
                            </select>
                        </div>
                        <button type="submit" class="btn btn-primary btn-block">Calculate Cost</button>
                    </form>
                    <div id="result" class="mt-4" style="display:none;">
                        <h3>Expected Cost : $<span id="predictionValue"></span></h3>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <script>
            document.getElementById('predictionForm').addEventListener('submit', function(event) {
                event.preventDefault();
                
                const age = document.getElementById('age').value;
                const sex = document.getElementById('sex').value;
                const bmi = document.getElementById('bmi').value;
                const children = document.getElementById('children').value;
                const smoker = document.getElementById('smoker').value;
                const region = document.getElementById('region').value;

                fetch('http://127.0.0.1:8000/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        age: parseInt(age),
                        sex: sex,
                        bmi: parseFloat(bmi),
                        children: parseInt(children),
                        smoker: smoker,
                        region: region
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('predictionValue').textContent = data.prediction.toFixed(2);
                    document.getElementById('result').style.display = 'block';
                })
                .catch((error) => {
                    console.error('Error:', error);
                });
            });
        </script>
    </body>
    </html>
    """)

@app.post("/predict", response_model=OutputModel)
def predict(data: InputModel):
    data_df = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data_df)
    return OutputModel(prediction=round(predictions["prediction"].iloc[0], 2))
    
    



