import yaml
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io

from sql_agent.db_chat import MultiDatabaseQueryWithDeepSeek

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class CustomSQLRequest(BaseModel):
    sql_query: str
    db_name: str

class ConfigUpdateRequest(BaseModel):
    databases: dict = None
    base_url: str = None

def load_config(config_file='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_file='config.yaml'):
    """Save configuration to YAML file"""
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# Load initial configuration
config = load_config()
DB_CONFIGS = config['databases']
BASE_URL = config['base_url']

# 创建查询对象，指定知识库文件路径
db_query = MultiDatabaseQueryWithDeepSeek(DB_CONFIGS,
                                          "sk-***",
                                          is_local=True,
                                          knowledge_base_file="knowledge_base.json",
                                          base_url=BASE_URL)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def query_db(request: QueryRequest):
    # 示例查询 - 指定数据库
    try:
        user_question = request.user_question
        response = db_query.query(user_question, db_name=request.db_name)
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"查询错误: {str(e)}")
        print(f"详细错误信息:\n{error_details}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "details": error_details
        }, status_code=500)

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
        print(f"接收到图表生成请求: {request}")
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
        import traceback
        error_details = traceback.format_exc()
        print(f"图表生成错误: {str(e)}")
        print(f"详细错误信息:\n{error_details}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "details": error_details
        }, status_code=500)

@app.post("/execute-sql")
def execute_custom_sql(request: CustomSQLRequest):
    """
    执行用户自定义的SQL查询
    """
    try:
        result = db_query.execute_custom_sql(
            sql_query=request.sql_query,
            db_name=request.db_name
        )
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"自定义SQL执行错误: {str(e)}")
        print(f"详细错误信息:\n{error_details}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "details": error_details
        }, status_code=500)

@app.get("/config")
def get_config():
    """
    获取当前配置
    """
    try:
        config = load_config()
        return {"success": True, "data": config}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.put("/config")
def update_config(request: ConfigUpdateRequest):
    """
    更新配置
    """
    try:
        # 加载现有配置
        config = load_config()
        
        # 更新配置项
        if request.databases is not None:
            config['databases'] = request.databases
            global DB_CONFIGS
            DB_CONFIGS = request.databases
        
        if request.base_url is not None:
            config['base_url'] = request.base_url
            global BASE_URL
            BASE_URL = request.base_url
        
        # 保存配置到文件
        save_config(config)
        
        # 更新db_query对象的配置
        global db_query
        db_query = MultiDatabaseQueryWithDeepSeek(DB_CONFIGS,
                                                 "sk-***",
                                                 is_local=True,
                                                 knowledge_base_file="knowledge_base.json",
                                                 base_url=BASE_URL)
        
        return {"success": True, "message": "配置更新成功", "data": config}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"配置更新错误: {str(e)}")
        print(f"详细错误信息:\n{error_details}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "details": error_details
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)