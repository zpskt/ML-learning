from fastapi import FastAPI

from sql_agent.db_chat import MultiDatabaseQueryWithDeepSeek

app = FastAPI()
from pydantic import BaseModel

class QueryRequest(BaseModel):
    user_question: str
    db_name: str

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)