from langchain_community.utilities.sql_database import SQLDatabase
from openai import OpenAI


class DeepSeekLLM:
    """
    DeepSeek LLM 类，用于调用 DeepSeek API 生成文本
    """

    def __init__(self, api_key, is_local=False, model=None):
        """
        初始化 DeepSeek LLM 客户端

        Args:
            api_key (str): DeepSeek API 密钥（本地部署时可传任意值）
            is_local (bool): 是否为本地部署
            model (str): 模型名称，如果为None则自动选择
        """
        # 设置模型名称
        if model is None:
            # Ollama 中的模型名称通常是 "deepseek-r1" 或类似的
            self.model = "deepseek-r1:7b" if is_local else "deepseek-chat"
        else:
            self.model = model

        # 创建 OpenAI 客户端实例，配置 API 的基础 URL
        if is_local:
            self.client = OpenAI(
                api_key=api_key or "ollama",  # 本地部署可传任意值
                base_url="http://localhost:11434/v1"  # Ollama 的 API 地址
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

    def invoke(self, prompt):
        """
        调用 API 生成响应

        Args:
            prompt (str): 发送给模型的提示词

        Returns:
            str: 模型生成的响应内容，如果出错则返回错误信息
        """
        try:
            # 调用 API 创建聊天完成
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的SQL助手，能够根据用户问题生成准确的SQL查询语句。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.1  # 对于 SQL 生成，使用较低的温度以获得更确定性的结果
            )
            # 返回模型生成的内容
            return response.choices[0].message.content
        except Exception as e:
            # 异常处理，返回错误信息
            return f"API调用失败: {str(e)}"


class MultiDatabaseQueryWithDeepSeek:
    """
    支持多数据库查询的类
    可以根据用户指定的数据库和自然语言问题自动生成 SQL 并执行查询
    """

    def __init__(self, db_configs, deepseek_api_key, is_local=False):
        """
        初始化多数据库查询器

        Args:
            db_configs (dict): 数据库配置字典，格式为 {'db_name': 'db_uri', ...}
            deepseek_api_key (str): DeepSeek API 密钥（本地部署时可传任意值）
            is_local (bool): 是否为本地部署
        """
        # 存储数据库配置
        self.db_configs = db_configs
        # 初始化数据库连接字典
        self.databases = {}
        # 初始化当前数据库名称
        self.current_db = None
        # 初始化 LLM 实例
        self.llm = DeepSeekLLM(deepseek_api_key, is_local)

    def connect_database(self, db_name):
        """
        连接到指定数据库

        Args:
            db_name (str): 数据库名称

        Returns:
            bool: 连接是否成功
        """
        if db_name not in self.db_configs:
            raise ValueError(f"数据库 '{db_name}' 未在配置中找到")
        
        try:
            # 连接数据库
            self.databases[db_name] = SQLDatabase.from_uri(self.db_configs[db_name])
            self.current_db = db_name
            return True
        except Exception as e:
            raise ConnectionError(f"连接数据库 '{db_name}' 失败: {str(e)}")

    def get_database_schema(self, db_name=None):
        """
        获取指定数据库的 schema 信息，包括表结构等
        这些信息将帮助模型理解数据库结构

        Args:
            db_name (str, optional): 数据库名称，如果为None则使用当前数据库

        Returns:
            str: 数据库表信息
        """
        if not db_name:
            db_name = self.current_db
            
        if not db_name or db_name not in self.databases:
            raise ValueError("未指定数据库或数据库未连接")
            
        return self.databases[db_name].get_table_info()

    def get_available_databases(self):
        """
        获取可用的数据库列表

        Returns:
            list: 可用数据库名称列表
        """
        return list(self.db_configs.keys())

    def generate_sql_prompt(self, user_question, db_name=None):
        """
        生成给模型的提示词，包含数据库结构和用户问题

        Args:
            user_question (str): 用户的自然语言问题
            db_name (str, optional): 数据库名称，如果为None则使用当前数据库

        Returns:
            str: 构造好的提示词
        """
        if not db_name:
            db_name = self.current_db
            
        # 获取数据库 schema 信息
        schema = self.get_database_schema(db_name)

        # 构造完整的提示词，告诉模型数据库结构和要求
        prompt = f"""
        你是一个SQL专家。请根据以下数据库结构和用户问题，生成准确的SQL查询语句。

        当前查询的数据库: {db_name}
        数据库结构：
        {schema}

        用户问题：{user_question}

        要求：
        1. 只返回SQL查询语句，不要有其他解释
        2. 确保SQL语法正确
        3. 使用合适的查询条件
        4. 如果用户问题不明确，请基于数据库结构做出合理假设

        SQL查询：
        """
        return prompt

    def query(self, user_question, db_name=None):
        """
        执行完整的查询流程：生成提示词 -> 调用LLM -> 执行SQL -> 返回结果

        Args:
            user_question (str): 用户的自然语言问题
            db_name (str, optional): 数据库名称，如果为None则使用当前数据库

        Returns:
            str: 格式化的查询结果，包含问题、SQL和执行结果
        """
        try:
            # 如果指定了数据库，则切换到该数据库
            if db_name:
                if db_name not in self.databases:
                    self.connect_database(db_name)
                self.current_db = db_name
            
            if not self.current_db:
                raise ValueError("未指定要查询的数据库")

            # 1. 生成提示词
            prompt = self.generate_sql_prompt(user_question, db_name)
            print("生成的提示词:", prompt[:500] + "..." if len(prompt) > 500 else prompt)  # 限制打印长度

            # 2. 调用模型生成SQL
            sql_query = self.llm.invoke(prompt)
            print(f"模型返回的原始结果: {sql_query}")

            # 3. 清理SQL结果（去除可能的额外文本）
            sql_query = self.clean_sql_query(sql_query)
            print(f"清理后的SQL: {sql_query}")

            # 4. 执行查询
            result = self.databases[self.current_db].run(sql_query)

            # 5. 生成自然语言回复
            response = self.generate_response(user_question, sql_query, result, self.current_db)
            return response

        except Exception as e:
            return f"查询过程中出错：{str(e)}"

    def clean_sql_query(self, sql_text):
        """
        清理SQL查询，去除模型可能添加的额外文本

        Args:
            sql_text (str): 模型返回的原始文本

        Returns:
            str: 提取出的纯净 SQL 查询语句
        """
        # 提取SQL代码块（如果有```sql ```包装）
        if "```sql" in sql_text:
            start = sql_text.find("```sql") + 6
            end = sql_text.find("```", start)
            sql_text = sql_text[start:end].strip()
        elif "```" in sql_text:
            start = sql_text.find("```") + 3
            end = sql_text.find("```", start)
            sql_text = sql_text[start:end].strip()
        
        # 移除多余的换行符和多余空格，将多行SQL合并为单行
        # 将多个连续的空白字符（包括换行符、制表符、空格）替换为单个空格
        import re
        cleaned_sql = re.sub(r'\s+', ' ', sql_text).strip()
        
        # 如果没有任何代码块标记，直接返回去除首尾空格的文本
        return cleaned_sql

    def generate_response(self, question, sql, result, db_name):
        """
        生成友好的自然语言回复

        Args:
            question (str): 用户的问题
            sql (str): 执行的 SQL 查询语句
            result (str): 查询结果
            db_name (str): 查询的数据库名称

        Returns:
            dict: 结构化的响应数据
        """
        return {
            "success": True,
            "data": {
                "database": db_name,
                "question": question,
                "sql": sql,
                "result": result
            },
            "error": None
        }

def main():
    """
    主函数，演示如何使用 MultiDatabaseQueryWithDeepSeek 类
    """
    # 示例用法
    # 从环境变量获取 API 密钥，如果没有设置则使用默认值
    DEEPSEEK_API_KEY = "sk-****"

    # 多个数据库连接信息
    # 格式: {'数据库名': '数据库URI', ...}
    DB_CONFIGS = {
        "cloud_platform": "mysql+pymysql://root:zhangpeng@localhost:3306/cloud_platform",
        "ecommerce_management": "mysql+pymysql://root:zhangpeng@localhost:3306/ecommerce_management",
    }

    # 创建查询对象
    db_query = MultiDatabaseQueryWithDeepSeek(DB_CONFIGS, DEEPSEEK_API_KEY, is_local=True)

    # 示例查询 - 指定数据库
    user_question = "有多少个用户使用？"
    response = db_query.query(user_question, db_name="ecommerce_management")
    print(response)

# 程序入口点
if __name__ == '__main__':
    main()