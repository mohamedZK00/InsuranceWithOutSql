import os
import pandas as pd
import joblib
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# إنشاء التطبيق
app = FastAPI()

# تحديد مسار الموديل
model_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(model_dir, '1-Insurance_model')

print(f"Loading model from: {model_path}")

# تحميل الموديل
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

# إعداد CORS
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

# الدالة الرئيسية لإرجاع صفحة HTML
@app.get("/", response_class=HTMLResponse)
def read_root():
    return FileResponse("index.html")

# دالة التنبؤ
@app.post("/predict", response_model=OutputModel)
def predict(data: InputModel):
    data_df = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data_df)
    return OutputModel(prediction= round(predictions['prediction_label'].iloc[0], 2))
