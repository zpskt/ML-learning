from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
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

class KnowledgeBaseEntry(BaseModel):
    question_template: str
    sql_query: str
    database: str
    description: str = ""

class KnowledgeBaseUpdate(BaseModel):
    id: int
    question_template: str = None
    sql_query: str = None
    database: str = None
    description: str = None

class KnowledgeBaseId(BaseModel):
    id: int

class ChartRequest(BaseModel):
    query_result: str
    chart_type: str  # 'bar', 'line', 'pie', 'scatter', 'histogram'
    title: str = ""
    x_label: str = ""
    y_label: str = ""


DB_CONFIGS = {
    "cloud_platform": "mysql+pymysql://root:zhangpeng@localhost:3306/cloud_platform",
    "storage": "mongodb://localhost:27017/storage"
}

# 创建查询对象，指定知识库文件路径
db_query = MultiDatabaseQueryWithDeepSeek(DB_CONFIGS, "sk-***", is_local=True, knowledge_base_file="knowledge_base.json")

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

# 知识库管理接口
@app.post("/knowledge")
def add_knowledge(entry: KnowledgeBaseEntry):
    """
    添加知识库条目
    """
    try:
        knowledge_entry = db_query.add_knowledge(
            question_template=entry.question_template,
            sql_query=entry.sql_query,
            database=entry.database,
            description=entry.description
        )
        return {"success": True, "data": knowledge_entry}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.put("/knowledge")
def update_knowledge(update_data: KnowledgeBaseUpdate):
    """
    更新知识库条目
    """
    try:
        knowledge_entry = db_query.update_knowledge(
            knowledge_id=update_data.id,
            question_template=update_data.question_template,
            sql_query=update_data.sql_query,
            database=update_data.database,
            description=update_data.description
        )
        if knowledge_entry:
            return {"success": True, "data": knowledge_entry}
        else:
            return {"success": False, "error": "Knowledge entry not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/knowledge")
def delete_knowledge(entry_id: KnowledgeBaseId):
    """
    删除知识库条目
    """
    try:
        result = db_query.delete_knowledge(entry_id.id)
        if result:
            return {"success": True, "message": "Knowledge entry deleted"}
        else:
            return {"success": False, "error": "Knowledge entry not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/knowledge")
def get_knowledge(knowledge_id: int = None):
    """
    获取知识库条目
    """
    try:
        if knowledge_id is not None:
            knowledge_entry = db_query.get_knowledge(knowledge_id)
            if knowledge_entry:
                return {"success": True, "data": knowledge_entry}
            else:
                return {"success": False, "error": "Knowledge entry not found"}
        else:
            knowledge_entries = db_query.get_knowledge()
            return {"success": True, "data": knowledge_entries}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/chart/generate")
def generate_chart(request: ChartRequest):
    """
    根据查询结果生成图表
    """
    try:
        chart_data = db_query.generate_chart(
            query_result=request.query_result,
            chart_type=request.chart_type,
            title=request.title,
            x_label=request.x_label,
            y_label=request.y_label
        )
        return JSONResponse(content={
            "success": True,
            "data": {
                "chart_type": request.chart_type,
                "image": chart_data  # base64编码的图片数据
            }
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)