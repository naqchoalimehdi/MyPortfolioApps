import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import os

app = FastAPI()

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2,
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

class ChatRequest(BaseModel):
    user_input1: str

@app.post("/tools/chatbot")
async def chat(request: ChatRequest):
    try:
        prompt = "you are helpful assistant who responds to {user_input} sweetly and simply"
        prompt_template = PromptTemplate.from_template(prompt)
        formatted_prompt = prompt_template.invoke({"user_input": request.user_input1}) 
        response = llm.invoke(formatted_prompt)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
