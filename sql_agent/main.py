from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import io

from sql_agent.db_chat import MultiDatabaseQueryWithDeepSeek

app = FastAPI()
from pydantic import BaseModel

class QueryRequest(BaseModel):
    user_question: str
    db_name: str

class HistoryRequest(BaseModel):
    limit: int = 10

class ExportRequest(BaseModel):
    query_result: str
    filename: str = "query_result.csv"

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

@app.post("/export/csv")
def export_to_csv(request: ExportRequest):
    """
    TODO 这里需要完善、添加联动
    将查询结果导出为 CSV 格式
    """
    try:
        csv_content = db_query.export_to_csv(request.query_result, request.filename)
        
        # 创建字节流
        stream = io.StringIO(csv_content)
        response = StreamingResponse(
            iter([stream.getvalue()]), 
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={request.filename}"}
        )
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)