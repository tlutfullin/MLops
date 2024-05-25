from datetime import datetime
import os
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.background import BackgroundTasks

from app.src import scorer, preprocessing 

app = FastAPI()

# Directory for storing output and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/upload", response_class=HTMLResponse)
async def upload_get(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})



@app.post("/upload")
async def upload_post(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if allowed_file(file.filename):
        filename = os.path.basename(file.filename)
        safe_filename = datetime.now().strftime(f"{filename.split('.')[0]}_%Y%m%d%H%M%S.csv")
        input_path = os.path.join('input', safe_filename)

        os.makedirs('input', exist_ok=True)  # Создание директории, если она не существует
        with open(input_path, "wb+") as file_object:
            file_object.write(await file.read())

        background_tasks.add_task(process_file, input_path)
        
        os.makedirs('output', exist_ok=True)  # Убедитесь, что директория output существует
        return templates.TemplateResponse("download.html", {"request": request, "files": os.listdir('output')})

    raise HTTPException(status_code=400, detail="Invalid file type")



async def process_file(file_path: str):
    input_df = preprocessing.import_data(file_path)
    preprocessed_df = preprocessing.run_preproc(input_df)
    submission = scorer.make_pred(preprocessed_df, file_path)
    submission.to_csv(file_path.replace('input', 'output'), index=False)



@app.get("/download", response_class=HTMLResponse)
async def download(request: Request):
    return templates.TemplateResponse("download.html", {"request": request, "files": os.listdir('output')})



@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(path=os.path.join('output', filename), filename=filename)
