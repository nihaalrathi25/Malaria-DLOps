from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.inference.predict import predict_image
import shutil
import os

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="app/templates")

# Static files (CSS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = predict_image(file_location)

    os.remove(file_location)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "image_name": file.filename
        },
    )