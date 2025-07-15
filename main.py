from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(title="Breast Cancer Classifier")

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model_path = "C:/Users/scan/projects/ML-project2/Logistic Regression Project/models/decision_tree_model.pkl"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

# Input schema
class CancerInput(BaseModel):
    mean_radius: float = Field(..., gt=0)
    mean_texture: float = Field(..., gt=0)
    mean_perimeter: float = Field(..., gt=0)
    mean_area: float = Field(..., gt=0)
    mean_smoothness: float = Field(..., gt=0)
    mean_compactness: float = Field(..., gt=0)
    mean_concavity: float = Field(..., gt=0)
    mean_concave_points: float = Field(..., gt=0)
    mean_symmetry: float = Field(..., gt=0)
    mean_fractal_dimension: float = Field(..., gt=0)
    radius_error: float = Field(..., gt=0)
    texture_error: float = Field(..., gt=0)
    perimeter_error: float = Field(..., gt=0)
    area_error: float = Field(..., gt=0)
    smoothness_error: float = Field(..., gt=0)
    compactness_error: float = Field(..., gt=0)
    concavity_error: float = Field(..., gt=0)
    concave_points_error: float = Field(..., gt=0)
    symmetry_error: float = Field(..., gt=0)
    fractal_dimension_error: float = Field(..., gt=0)
    worst_radius: float = Field(..., gt=0)
    worst_texture: float = Field(..., gt=0)
    worst_perimeter: float = Field(..., gt=0)
    worst_area: float = Field(..., gt=0)
    worst_smoothness: float = Field(..., gt=0)
    worst_compactness: float = Field(..., gt=0)
    worst_concavity: float = Field(..., gt=0)
    worst_concave_points: float = Field(..., gt=0)
    worst_symmetry: float = Field(..., gt=0)
    worst_fractal_dimension: float = Field(..., gt=0)

# API root
@app.get("/")
def read_root():
    return {"message": "Welcome to Breast Cancer Prediction API"}

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# API JSON prediction
@app.post("/predict")
def predict(cancer: CancerInput):
    try:
        features = np.array([[value for value in cancer.dict().values()]])
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        result = ["malignant", "benign"][prediction]
        return {
            "prediction": result,
            "probability": prediction_proba.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Web form
@app.get("/form", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle form POST
@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(request: Request,
                 mean_radius: float = Form(...),
                 mean_texture: float = Form(...),
                 mean_perimeter: float = Form(...),
                 mean_area: float = Form(...),
                 mean_smoothness: float = Form(...),
                 mean_compactness: float = Form(...),
                 mean_concavity: float = Form(...),
                 mean_concave_points: float = Form(...),
                 mean_symmetry: float = Form(...),
                 mean_fractal_dimension: float = Form(...),
                 radius_error: float = Form(...),
                 texture_error: float = Form(...),
                 perimeter_error: float = Form(...),
                 area_error: float = Form(...),
                 smoothness_error: float = Form(...),
                 compactness_error: float = Form(...),
                 concavity_error: float = Form(...),
                 concave_points_error: float = Form(...),
                 symmetry_error: float = Form(...),
                 fractal_dimension_error: float = Form(...),
                 worst_radius: float = Form(...),
                 worst_texture: float = Form(...),
                 worst_perimeter: float = Form(...),
                 worst_area: float = Form(...),
                 worst_smoothness: float = Form(...),
                 worst_compactness: float = Form(...),
                 worst_concavity: float = Form(...),
                 worst_concave_points: float = Form(...),
                 worst_symmetry: float = Form(...),
                 worst_fractal_dimension: float = Form(...)):
    try:
        features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area,
                              mean_smoothness, mean_compactness, mean_concavity,
                              mean_concave_points, mean_symmetry, mean_fractal_dimension,
                              radius_error, texture_error, perimeter_error, area_error,
                              smoothness_error, compactness_error, concavity_error,
                              concave_points_error, symmetry_error, fractal_dimension_error,
                              worst_radius, worst_texture, worst_perimeter, worst_area,
                              worst_smoothness, worst_compactness, worst_concavity,
                              worst_concave_points, worst_symmetry, worst_fractal_dimension]])
        prediction = model.predict(features)[0]
        result = ["malignant", "benign"][prediction]
        return templates.TemplateResponse("index.html", {"request": request, "prediction": result})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "prediction": f"Error: {e}"})
