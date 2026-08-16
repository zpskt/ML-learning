#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：knowledge_base.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/12 22:12 
'''
import hashlib
import os
from datetime import datetime
from importlib.metadata import metadata

import config_data as config

def get_string_md5(input_str:str,encoding='utf-8'):
    #将字符串转bytes字节数组
    str_bytes = input_str.encode(encoding)

    #创建md5对象
    hash_md5 = hashlib.md5(str_bytes).hexdigest()
    return hash_md5

def check_md5_exist(md5_hash:str)-> bool:
    if not os.path.exists(config.md5_path):
#         不存在，创建
        open(config.md5_path,'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip() #处理字符串前后的空格和回车
            if md5_hash in line:
                return True
        return False



# 知识库
class KnowledgeBaseService(object):
    def __init__(self):
        #创建默认文件夹,如果存在则跳过
        os.makedirs(config.persist_dir, exist_ok=True)
        self.chroma = None
        self.spliter = None

    def upload_by_string(self, data:str,filename:str):
        """将传入的字符串，进行向量化，存入向量数据库汇总"""

        # 判断是否已经有此内容
        if check_md5_exist(get_string_md5(data)):
            return "[跳过]内容已经存在知识库中"
        if len(data)> config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]
        metadata = {
            "source": filename,
            "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "creator": "张鹏"
        }




if __name__ == '__main__':
    input_str  ="123"
    md_= get_string_md5(input_str)
    print(input_str)
    print(md_)