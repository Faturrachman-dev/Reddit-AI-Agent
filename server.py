from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

class BaseClass(BaseModel):
    task: str

class Todo(BaseModel):
    id: Optional[int]
    task: str
    is_completed: bool = False

class ReturnTodo(BaseClass):
    pass

app = FastAPI()
todos = []

@app.post('/todos', response_model=ReturnTodo)
async def add_todo(data: Todo):
    data.id = len(todos)+1
    todos.append(data)
    return data

@app.get('/todos', response_model=ReturnTodo)
async def get_todos():
    return todos

@app.get('/todos/{id}')
async def get_todo(id: int):
    for todo in todos:
        if(id==todo.id):
            return todo
    raise HTTPException(status_code=404, detail="Item not found")

@app.put('/todos/{id}')
async def update_todo(id: int, data: Todo):
    for index, todo in enumerate(todos):
        if(todo.id==id):
            data.id = len(todos)+1
            todos[index] = data
            return
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete('/todos/{id}')
async def delete_todo(id: int):
    for index, todo in enumerate(todos):
        if(todo.id==id):
            del todos[index]
            return
    raise HTTPException(status_code=404, detail="Item not found")

# uvicorn server:app --host 127.0.0.1 --port 5566 --reload