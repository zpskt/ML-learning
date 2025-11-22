from langchain_community.utilities import SQLDatabase
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


class DatabaseQueryWithDeepSeek:
    """
    结合 DeepSeek LLM 和数据库的查询类
    可以根据自然语言问题自动生成 SQL 并执行查询
    """

    def __init__(self, db_uri, deepseek_api_key, is_local=False):
        """
        初始化数据库查询器

        Args:
            db_uri (str): 数据库连接 URI
            deepseek_api_key (str): DeepSeek API 密钥（本地部署时可传任意值）
            is_local (bool): 是否为本地部署
        """
        # 连接数据库
        self.db = SQLDatabase.from_uri(db_uri)
        # 初始化 LLM 实例
        self.llm = DeepSeekLLM(deepseek_api_key, is_local)

    def get_database_schema(self):
        """
        获取数据库 schema 信息，包括表结构等
        这些信息将帮助模型理解数据库结构

        Returns:
            str: 数据库表信息
        """
        return self.db.get_table_info()

    def generate_sql_prompt(self, user_question):
        """
        生成给模型的提示词，包含数据库结构和用户问题

        Args:
            user_question (str): 用户的自然语言问题

        Returns:
            str: 构造好的提示词
        """
        # 获取数据库 schema 信息
        schema = self.get_database_schema()

        # 构造完整的提示词，告诉模型数据库结构和要求
        prompt = f"""
        你是一个SQL专家。请根据以下数据库结构和用户问题，生成准确的SQL查询语句。

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

    def query(self, user_question):
        """
        执行完整的查询流程：生成提示词 -> 调用LLM -> 执行SQL -> 返回结果

        Args:
            user_question (str): 用户的自然语言问题

        Returns:
            str: 格式化的查询结果，包含问题、SQL和执行结果
        """
        try:
            # 1. 生成提示词
            prompt = self.generate_sql_prompt(user_question)
            print("生成的提示词:", prompt[:500] + "..." if len(prompt) > 500 else prompt)  # 限制打印长度

            # 2. 调用模型生成SQL
            sql_query = self.llm.invoke(prompt)
            print(f"模型返回的原始结果: {sql_query}")

            # 3. 清理SQL结果（去除可能的额外文本）
            sql_query = self.clean_sql_query(sql_query)
            print(f"清理后的SQL: {sql_query}")

            # 4. 执行查询
            result = self.db.run(sql_query)

            # 5. 生成自然语言回复
            response = self.generate_response(user_question, sql_query, result)
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
            return sql_text[start:end].strip()
        elif "```" in sql_text:
            start = sql_text.find("```") + 3
            end = sql_text.find("```", start)
            return sql_text[start:end].strip()
        else:
            # 如果没有任何代码块标记，直接返回去除首尾空格的文本
            return sql_text.strip()

    def generate_response(self, question, sql, result):
        """
        生成友好的自然语言回复

        Args:
            question (str): 用户的问题
            sql (str): 执行的 SQL 查询语句
            result (str): 查询结果

        Returns:
            str: 格式化的响应文本
        """
        return f"""
                🤖 **查询结果**
                
                **您的问题**：{question}
                
                **生成的SQL**：
                ```sql
                {sql}
                ```
                
                **查询结果**：
                {result}
                            
                """

def main():
    """
    主函数，演示如何使用 DatabaseQueryWithDeepSeek 类
    """
    # 示例用法
    # 从环境变量获取 API 密钥，如果没有设置则使用默认值
    DEEPSEEK_API_KEY = "sk-****"

    # MySQL数据库连接信息
    # 格式: mysql+pymysql://用户名:密码@主机:端口/数据库名
    DB_URI = "mysql+pymysql://root:zhangpeng@localhost:3306/cloud_platform"

    # 创建查询对象
    db_query = DatabaseQueryWithDeepSeek(DB_URI, DEEPSEEK_API_KEY, is_local=True)

    # 示例查询
    user_question = "移动应用开发这个项目现在是由谁负责跟进？"
    response = db_query.query(user_question)
    print(response)


# 程序入口点
if __name__ == '__main__':
    main()
