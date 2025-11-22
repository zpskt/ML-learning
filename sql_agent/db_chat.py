import csv
import io
from langchain_community.utilities.sql_database import SQLDatabase
from openai import OpenAI
import datetime
import json
import re
import os


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

    def __init__(self, db_configs, deepseek_api_key, is_local=False, knowledge_base_file="knowledge_base.json"):
        """
        初始化多数据库查询器

        Args:
            db_configs (dict): 数据库配置字典，格式为 {'db_name': 'db_uri', ...}
            deepseek_api_key (str): DeepSeek API 密钥（本地部署时可传任意值）
            is_local (bool): 是否为本地部署
            knowledge_base_file (str): 知识库文件路径
        """
        # 存储数据库配置
        self.db_configs = db_configs
        # 初始化数据库连接字典
        self.databases = {}
        # 初始化当前数据库名称
        self.current_db = None
        # 初始化 LLM 实例
        self.llm = DeepSeekLLM(deepseek_api_key, is_local)
        # 初始化查询历史记录列表
        self.query_history = []
        # 初始化知识库文件路径
        self.knowledge_base_file = knowledge_base_file
        # 初始化知识库
        self.knowledge_base = self._load_knowledge_base()

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
        你是一个数据库专家，能够根据数据库结构和用户问题，生成准确的SQL查询语句.

        当前查询的数据库: {db_name}
        数据库结构：
        {schema}

        用户问题：{user_question}

        请按照以下格式回答：
        ```sql
        [在这里写入SQL查询语句]
        ```
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

            # 记录查询开始时间
            start_time = datetime.datetime.now()

            # 1. 首先尝试在知识库中查找匹配的查询
            knowledge_entry = self.search_knowledge(user_question, db_name)
            
            if knowledge_entry:
                # 如果在知识库中找到了匹配项，直接使用预定义的SQL
                sql_query = knowledge_entry["sql_query"]
                print(f"从知识库中找到匹配项: {knowledge_entry['question_template']}")
            else:
                # 如果知识库中没有匹配项，使用原来的LLM生成方式
                # 生成提示词
                prompt = self.generate_sql_prompt(user_question, db_name)
                print("生成的提示词:", prompt[:500] + "..." if len(prompt) > 500 else prompt)  # 限制打印长度

                # 调用模型生成SQL和自然语言回答
                llm_response = self.llm.invoke(prompt)
                print(f"模型返回的原始结果: {llm_response}")

                # 提取纯净的sql查询
                sql_query = self.clean_sql_query(llm_response)
            
            print(f"解析后的SQL: {sql_query}")

            # 执行查询
            result = self.databases[self.current_db].run(sql_query)

            # 解析SQL语句，提取表名
            table_names = self._extract_table_names(sql_query)
            # 生成自然语言回答
            natural_response = self._generate_natural_response(user_question, result, table_names)
            # 返回结果
            response = {
                "success": True,
                "data": {
                    "database": self.current_db,
                    "table_names": table_names,
                    "question": user_question,
                    "sql": sql_query,
                    "result": result,
                    "natural_response": natural_response,
                    "from_knowledge_base": knowledge_entry is not None
                },
                "error": None
            }
            
            # 记录查询结束时间
            end_time = datetime.datetime.now()
            
            # 添加到查询历史
            self.add_to_history({
                "timestamp": start_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "question": user_question,
                "database": self.current_db,
                "sql": sql_query,
                "result": result,
                "success": True,
                "error": None,
                "from_knowledge_base": knowledge_entry is not None
            })

            return response

        except Exception as e:
            # 记录错误信息到历史
            error_time = datetime.datetime.now()
            self.add_to_history({
                "timestamp": error_time.isoformat(),
                "duration": 0,
                "question": user_question,
                "database": db_name or self.current_db,
                "sql": "",
                "result": "",
                "success": False,
                "error": str(e)
            })
            return f"查询过程中出错：{str(e)}"

    def add_to_history(self, record):
        """
        添加查询记录到历史列表

        Args:
            record (dict): 查询记录
        """
        self.query_history.append(record)
        # 限制历史记录数量，最多保留100条记录
        if len(self.query_history) > 100:
            self.query_history.pop(0)  # 删除最旧的记录

    def get_query_history(self, limit=10):
        """
        获取查询历史记录

        Args:
            limit (int): 返回记录的数量限制，默认为10条

        Returns:
            list: 查询历史记录列表
        """
        # 返回最近的limit条记录，按时间倒序排列
        return self.query_history[-limit:][::-1]

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

    def export_to_csv(self, query_result, filename="query_result.csv"):
        """
        将查询结果导出为 CSV 格式
        
        Args:
            query_result (str): 数据库查询结果字符串
            filename (str): 导出文件名
            
        Returns:
            str: CSV 格式的数据
        """
        try:
            # 解析查询结果
            parsed_result = self._parse_query_result(query_result)
            
            # 创建内存中的 CSV 文件
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 写入数据
            if parsed_result["headers"]:
                writer.writerow(parsed_result["headers"])
                
            for row in parsed_result["rows"]:
                writer.writerow(row)
                
            # 获取 CSV 字符串
            csv_content = output.getvalue()
            output.close()
            
            return csv_content
        except Exception as e:
            raise Exception(f"导出CSV失败: {str(e)}")
    
    def _parse_query_result(self, query_result):
        """
        解析查询结果字符串为结构化数据
        
        Args:
            query_result (str): 数据库查询结果字符串
            
        Returns:
            dict: 包含表头和行数据的字典
        """
        try:
            # 尝试将字符串形式的结果转换为Python对象
            import ast
            parsed_result = ast.literal_eval(query_result)
            
            # 如果结果是列表
            if isinstance(parsed_result, list):
                if len(parsed_result) == 0:
                    return {"headers": [], "rows": []}
                    
                # 如果是包含元组的列表（常见于数据库查询结果）
                if isinstance(parsed_result[0], tuple):
                    # 假设第一行是列标题（这只是一个简单的假设）
                    # 在实际应用中，我们可能需要从查询中提取真实的列名
                    headers = [f"Column_{i}" for i in range(len(parsed_result[0]))] if parsed_result else []
                    rows = [list(row) for row in parsed_result]
                    return {"headers": headers, "rows": rows}
                # 如果是简单列表
                else:
                    headers = ["Value"]
                    rows = [[item] for item in parsed_result]
                    return {"headers": headers, "rows": rows}
            else:
                # 处理单个值的情况
                headers = ["Result"]
                rows = [[parsed_result]]
                return {"headers": headers, "rows": rows}
                
        except (ValueError, SyntaxError):
            # 如果解析失败，将整个结果作为一个单元格返回
            return {"headers": ["Result"], "rows": [[query_result]]}
    
    def _extract_table_names(self, sql_query):
        """
        从SQL查询中提取表名

        Args:
            sql_query (str): SQL查询语句

        Returns:
            list: 表名列表
        """
        import re
        # 匹配 FROM 和 JOIN 后面的表名
        table_pattern = re.compile(r'\b(?:FROM|JOIN)\s+([\w"][\w\d$_]*[\w"]*(?:\.[\w"][\w\d$_]*[\w"]*)?)', re.IGNORECASE)
        matches = table_pattern.findall(sql_query)
        
        # 清理表名（去除可能的引号）
        table_names = []
        for table in matches:
            # 去除可能存在的双引号
            clean_table = table.replace('"', '')
            # 如果有别名（如 table AS alias），只保留表名
            if ' ' in clean_table:
                clean_table = clean_table.split()[0]
            table_names.append(clean_table)
        
        # 去重并保持顺序
        unique_tables = []
        for table in table_names:
            if table not in unique_tables:
                unique_tables.append(table)
        
        return unique_tables

    def parse_llm_response(self, llm_response):
        """
        解析模型的响应，提取SQL查询和自然语言回答

        Args:
            llm_response (str): 模型的完整响应

        Returns:
            tuple: (sql_query, natural_response) SQL查询和自然语言回答
        """
        import re
        
        # 提取SQL代码块
        sql_match = re.search(r"```sql\s*([\s\S]*?)\s*```", llm_response, re.IGNORECASE)
        sql_query = sql_match.group(1).strip() if sql_match else ""
        
        # 提取自然语言回答
        natural_match = re.search(r"自然语言回答[：:]\s*([\s\S]*)", llm_response, re.IGNORECASE)
        natural_response = natural_match.group(1).strip() if natural_match else ""
        
        # 清理SQL查询
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        return sql_query, natural_response

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
        # 解析SQL语句，提取表名
        table_names = self._extract_table_names(sql)

        # 生成自然语言回答
        natural_language_response = self._generate_natural_response(question, result, table_names)

        return {
            "success": True,
            "data": {
                "database": db_name,
                "table_names": table_names,
                "question": question,
                "sql": sql,
                "result": result,
                "natural_response": natural_language_response
            },
            "error": None
        }

    def _generate_natural_response(self, question, result, table_names):
        """
        生成自然语言回答

        Args:
            question (str): 用户问题
            result (str): 查询结果
            table_names (list): 表名列表

        Returns:
            str: 自然语言回答
        """
        # 解析查询结果
        try:
            # 尝试将字符串形式的结果转换为Python对象
            import ast
            parsed_result = ast.literal_eval(result)
            
            # 如果结果是非空列表
            if isinstance(parsed_result, list) and len(parsed_result) > 0:
                # 如果是包含元组的列表（常见于数据库查询结果）
                if isinstance(parsed_result[0], tuple):
                    # 提取第一个元组的第一个元素作为计数结果
                    count = parsed_result[0][0] if len(parsed_result[0]) > 0 else 0
                # 如果是简单列表
                elif isinstance(parsed_result[0], (int, float)):
                    count = parsed_result[0]
                else:
                    count = len(parsed_result)
            else:
                count = 0
                
        except (ValueError, SyntaxError):
            # 如果解析失败，使用默认方式处理
            count = "unknown"

        # 构造提示词让模型生成自然语言回答
        prompt = f"""
        你是一个数据库查询助手。用户询问了一个问题，系统已经从数据库中查询到了结果。
        请根据以下信息生成一个自然、易懂的回答：

        用户问题：{question}
        查询涉及的表：{', '.join(table_names) if table_names else '未知表'}
        查询结果：{result}
        解析后的数值结果：{count}

        要求：
        1. 回答应该简洁明了，使用自然语言
        2. 不要暴露技术细节如表名、字段名等
        3. 直接回答用户问题，不需要额外解释
        4. 如果结果为空或无意义，请给出合适的说明
        5. 在回答中应准确反映解析后的数值结果
        6. 告诉用户，主要是从什么表中获取的结果
        
        自然语言回答：
        """

        # 调用模型生成自然语言回答
        natural_response = self.llm.invoke(prompt)
        
        # 清理模型生成的文本
        if natural_response.startswith("自然语言回答："):
            natural_response = natural_response[len("自然语言回答："):].strip()
        
        return natural_response
    
    def _load_knowledge_base(self):
        """
        从文件加载知识库
        
        Returns:
            list: 知识库条目列表
        """
        if os.path.exists(self.knowledge_base_file):
            try:
                with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载知识库文件时出错: {e}")
                return []
        else:
            # 如果文件不存在，创建一个空的知识库文件
            self._save_knowledge_base([])
            return []
    
    def _save_knowledge_base(self, knowledge_base):
        """
        保存知识库到文件
        
        Args:
            knowledge_base (list): 知识库条目列表
        """
        try:
            with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存知识库文件时出错: {e}")
    
    def add_knowledge(self, question_template, sql_query, database, description=""):
        """
        添加知识库条目
        
        Args:
            question_template (str): 问题模板
            sql_query (str): 对应的SQL查询
            database (str): 适用的数据库
            description (str): 描述信息
            
        Returns:
            dict: 添加的知识条目
        """
        knowledge_entry = {
            "id": len(self.knowledge_base) + 1,
            "question_template": question_template,
            "sql_query": sql_query,
            "database": database,
            "description": description,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        self.knowledge_base.append(knowledge_entry)
        self._save_knowledge_base(self.knowledge_base)
        return knowledge_entry
    
    def update_knowledge(self, knowledge_id, question_template=None, sql_query=None, database=None, description=None):
        """
        更新知识库条目
        
        Args:
            knowledge_id (int): 知识条目ID
            question_template (str, optional): 问题模板
            sql_query (str, optional): 对应的SQL查询
            database (str, optional): 适用的数据库
            description (str, optional): 描述信息
            
        Returns:
            dict: 更新后的知识条目，如果未找到返回None
        """
        for entry in self.knowledge_base:
            if entry["id"] == knowledge_id:
                if question_template is not None:
                    entry["question_template"] = question_template
                if sql_query is not None:
                    entry["sql_query"] = sql_query
                if database is not None:
                    entry["database"] = database
                if description is not None:
                    entry["description"] = description
                entry["updated_at"] = datetime.datetime.now().isoformat()
                self._save_knowledge_base(self.knowledge_base)
                return entry
        return None
    
    def delete_knowledge(self, knowledge_id):
        """
        删除知识库条目
        
        Args:
            knowledge_id (int): 知识条目ID
            
        Returns:
            bool: 删除是否成功
        """
        for i, entry in enumerate(self.knowledge_base):
            if entry["id"] == knowledge_id:
                del self.knowledge_base[i]
                # 更新后续条目的ID
                for j in range(i, len(self.knowledge_base)):
                    self.knowledge_base[j]["id"] = j + 1
                self._save_knowledge_base(self.knowledge_base)
                return True
        return False
    
    def get_knowledge(self, knowledge_id=None):
        """
        获取知识库条目
        
        Args:
            knowledge_id (int, optional): 知识条目ID，如果为None则返回所有条目
            
        Returns:
            dict or list: 知识条目或条目列表
        """
        if knowledge_id is not None:
            for entry in self.knowledge_base:
                if entry["id"] == knowledge_id:
                    return entry
            return None
        return self.knowledge_base
    
    def search_knowledge(self, user_question, database=None):
        """
        在知识库中搜索匹配的条目
        
        Args:
            user_question (str): 用户问题
            database (str, optional): 指定数据库
            
        Returns:
            dict: 最佳匹配的知识条目，如果没有匹配项返回None
        """
        best_match = None
        best_score = 0
        
        for entry in self.knowledge_base:
            # 如果指定了数据库，且条目不适用于该数据库，则跳过
            if database and entry["database"] != database:
                continue
                
            # 计算匹配度 - 简单实现，可以根据需要改进
            score = self._calculate_similarity(user_question, entry["question_template"])
            
            if score > best_score:
                best_score = score
                best_match = entry
        
        # 设定一个阈值，只有匹配度足够高才返回
        if best_score > 0.5:
            return best_match
        return None
    
    def _calculate_similarity(self, question1, question2):
        """
        计算两个问题之间的相似度（简化实现）
        
        Args:
            question1 (str): 问题1
            question2 (str): 问题2
            
        Returns:
            float: 相似度得分 (0-1)
        """
        # 转为小写并移除标点符号
        q1 = re.sub(r'[^\w\s]', '', question1.lower())
        q2 = re.sub(r'[^\w\s]', '', question2.lower())
        
        # 分割为词汇集合
        words1 = set(q1.split())
        words2 = set(q2.split())
        
        # 计算交集和并集
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # 计算Jaccard相似度
        if len(union) == 0:
            return 1.0 if len(intersection) == 0 else 0.0
        return len(intersection) / len(union)

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