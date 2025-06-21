from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from backend.model import CurriculumScorer

app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Curriculum(BaseModel):
    id: int
    content: str
    score: float = 0.0
    explanation: str = ""

# In-memory database substitute
curriculums_db = []
scorer = CurriculumScorer()

@app.post("/upload-curriculum/")
async def upload_curriculum(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    curriculum = Curriculum(id=len(curriculums_db)+1, content=text)
    # Call AI model to score and explain
    curriculum.score = scorer.score(text)
    curriculum.explanation = scorer.explain(text)
    curriculums_db.append(curriculum)
    return {"message": "Curriculum uploaded and scored", "curriculum": curriculum}

@app.get("/curriculums/", response_model=List[Curriculum])
def get_curriculums():
    return curriculums_db

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
