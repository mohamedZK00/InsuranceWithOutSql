

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # تغيير إلى BaseModel لنماذج الإدخال والإخراج
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# إنشاء التطبيق
app = FastAPI()



# تحميل نموذج التدريبات
model = load_model("insurance_Model")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

# تعريف نماذج Pydantic للإدخال والإخراج
class InputModel(BaseModel):  # استخدام BaseModel بدلاً من create_model
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class OutputModel(BaseModel):  # استخدام BaseModel لنموذج الإخراج
    prediction: float



app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()




# تعريف دالة التنبؤ
@app.post("/predict", response_model=OutputModel)

def predict(data: InputModel):
    
    data_df = pd.DataFrame([data.dict()])  # تحويل الإدخال إلى DataFrame
    predictions = predict_model(model, data=data_df)
    return OutputModel(prediction=predictions["prediction_label"].iloc[0])  # إرجاع OutputModel

    
    
    
    
    



