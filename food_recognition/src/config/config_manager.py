"""
配置管理模块
用于管理食材识别项目的各种配置，包括事件监听器配置

作者: zhangpeng
时间: 2025-08-28
"""

import json
import os
import logging
from typing import Dict, List, Any


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为None，使用默认配置
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """
        获取默认配置文件路径
        
        Returns:
            str: 默认配置文件路径
        """
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, 'config.json')
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict: 配置字典
        """
        # 默认配置
        default_config = {
            "event_handlers": {
                "log": True,
                "tts": False,
                "api": False,
                "kafka": False
            },
            "tts": {
                "enabled": False,
                "rate": 200,
                "voice": "default"
            },
            "api": {
                "enabled": False,
                "endpoint": ""
            },
            "kafka": {
                "enabled": False,
                "bootstrap_servers": ["localhost:9092"],
                "topic": "food-detection-events"
            },
            "logging": {
                "enabled": True,
                "file": "logs/food_detection.log",
                "level": "INFO",
                "max_bytes": 10485760,
                "backup_count": 5
            }
        }
        
        # 如果配置文件存在，则加载配置文件
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # 合并配置，文件配置优先
                    self._merge_config(default_config, file_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
        
        return default_config
    
    def _merge_config(self, default: Dict, override: Dict) -> None:
        """
        合并配置字典
        
        Args:
            default: 默认配置
            override: 覆盖配置
        """
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key_path: str, default=None):
        """
        获取配置项
        
        Args:
            key_path: 配置项路径，如 "event_handlers.log"
            default: 默认值
            
        Returns:
            配置项值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def is_event_handler_enabled(self, handler_name: str) -> bool:
        """
        检查事件处理器是否启用
        
        Args:
            handler_name: 事件处理器名称 (log, tts, api, kafka)
            
        Returns:
            bool: 是否启用
        """
        return self.get(f"event_handlers.{handler_name}", False)
    
    def is_logging_enabled(self) -> bool:
        """
        检查日志记录是否启用
        
        Returns:
            bool: 是否启用
        """
        return self.get("logging.enabled", True)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            Dict: 日志配置
        """
        return {
            "file": self.get("logging.file", "logs/food_detection.log"),
            "level": self.get("logging.level", "INFO"),
            "max_bytes": self.get("logging.max_bytes", 10485760),
            "backup_count": self.get("logging.backup_count", 5)
        }
    
    def setup_logging(self):
        """
        根据配置设置日志记录
        """
        if not self.is_logging_enabled():
            return
        
        logging_config = self.get_logging_config()
        
        # 将级别字符串转换为logging模块常量
        level = getattr(logging, logging_config["level"].upper(), logging.INFO)
        
        # 确保日志目录存在
        log_file = logging_config["file"]
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建带轮转的文件处理器
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=logging_config["max_bytes"],
            backupCount=logging_config["backup_count"],
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # 获取记录器并添加处理器
        logger = logging.getLogger("food_detection")
        logger.setLevel(level)
        logger.addHandler(file_handler)
        
        # 同时也为root logger添加处理器
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        if not any(isinstance(handler, RotatingFileHandler) for handler in root_logger.handlers):
            root_logger.addHandler(file_handler)
    
    def save_config(self, config_path: str = None) -> None:
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径，默认使用实例路径
        """
        save_path = config_path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")


# 全局配置管理器实例
config_manager = ConfigManager()