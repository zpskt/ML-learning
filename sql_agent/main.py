from fastapi import FastAPI

from sql_agent.db_chat import MultiDatabaseQueryWithDeepSeek

app = FastAPI()
from pydantic import BaseModel

class QueryRequest(BaseModel):
    user_question: str
    db_name: str

class HistoryRequest(BaseModel):
    limit: int = 10

DB_CONFIGS = {
    "cloud_platform": "mysql+pymysql://root:zhangpeng@localhost:3306/cloud_platform",
    "ecommerce_management": "mysql+pymysql://root:zhangpeng@localhost:3306/ecommerce_management",
}

# 创建查询对象
db_query = MultiDatabaseQueryWithDeepSeek(DB_CONFIGS, "sk-***", is_local=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def query_db(request: QueryRequest):
    # 示例查询 - 指定数据库
    user_question = request.user_question
    response = db_query.query(user_question, db_name=request.db_name)
    
    return response

@app.post("/history")
def get_history(request: HistoryRequest):
    """
    获取查询历史记录
    """
    history = db_query.get_query_history(limit=request.limit)
    return {"success": True, "data": history}

@app.get("/history")
def get_history_get(limit: int = 10):
    """
    通过GET请求获取查询历史记录
    """
    history = db_query.get_query_history(limit=limit)
    return {"success": True, "data": history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)