from rag import generate_answer
from fastapi import FastAPI
from pydantic import BaseModel,Field,field_validator

app=FastAPI(
    title="RAG_API",
    description="Ask questions about DemoMart policies using RAG.",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question:str=Field(
        min_length=1,
        description="The question about DemoMart policies"
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls,value:str)->str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()

class SourceModel(BaseModel):
    file:str
    page:int

class AnswerResponse(BaseModel):
    answer:str
    sources:list[SourceModel]

@app.get("/health")
def health_check():
    return {"status":"ok"}

@app.post("/ask",response_model=AnswerResponse)
def ask_question(request:QuestionRequest):
    return generate_answer(request.question)