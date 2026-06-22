from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import os
import subprocess
import json

app = FastAPI()

#Connect api to frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE = "./workspace"
os.makedirs(WORKSPACE, exist_ok=True)

class Message(BaseModel):
    role: str   
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

