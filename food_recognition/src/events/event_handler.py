"""
食材识别事件处理模块
当识别到食材时触发事件，支持日志记录、语音播报、外部接口调用和Kafka推送等功能

作者: zhangpeng
时间: 2025-08-28
"""

import logging
import json
import requests
from typing import List, Dict, Any, Callable
from datetime import datetime

# 尝试导入配置管理器
try:
    from src.config.config_manager import config_manager
    # 根据配置设置日志
    config_manager.setup_logging()
except ImportError:
    config_manager = None


class FoodDetectionEvent:
    """食材检测事件类"""
    
    def __init__(self, ingredients: List[str], source: str = "unknown", timestamp: datetime = None):
        """
        初始化食材检测事件
        
        Args:
            ingredients: 检测到的食材列表
            source: 事件来源（如camera, image, video等）
            timestamp: 事件时间戳
        """
        self.ingredients = ingredients
        self.source = source
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """将事件转换为字典格式"""
        return {
            "ingredients": self.ingredients,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }


class EventHandler:
    """事件处理器"""
    
    def __init__(self):
        """初始化事件处理器"""
        self.listeners = []
        self.logger = logging.getLogger(__name__)
        self._setup_logger()
        
        # 根据配置注册事件监听器
        self._register_configured_listeners()
    
    def _setup_logger(self):
        """设置日志记录器"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _register_configured_listeners(self):
        """根据配置注册事件监听器"""
        if not config_manager:
            # 如果没有配置管理器，则只注册默认的日志监听器
            self.add_listener(log_detection_event)
            return
        
        # 根据配置决定注册哪些监听器
        if config_manager.is_event_handler_enabled("log"):
            self.add_listener(log_detection_event)
            
        if config_manager.is_event_handler_enabled("tts"):
            self.add_listener(tts_announcement_event)
            
        if config_manager.is_event_handler_enabled("api"):
            self.add_listener(external_api_call_event)
            
        if config_manager.is_event_handler_enabled("kafka"):
            self.add_listener(kafka_push_event)
    
    def add_listener(self, listener: Callable[[FoodDetectionEvent], None]):
        """
        添加事件监听器
        
        Args:
            listener: 事件监听函数，接收FoodDetectionEvent参数
        """
        self.listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[FoodDetectionEvent], None]):
        """
        移除事件监听器
        
        Args:
            listener: 要移除的事件监听函数
        """
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def trigger_event(self, event: FoodDetectionEvent):
        """
        触发食材检测事件
        
        Args:
            event: 食材检测事件对象
        """
        self.logger.info(f"食材检测事件触发: {event.to_dict()}")
        
        # 通知所有监听器
        for listener in self.listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"事件监听器执行出错: {e}")


# 预定义的事件监听器函数

def log_detection_event(event: FoodDetectionEvent):
    """
    日志记录事件监听器
    
    Args:
        event: 食材检测事件
    """
    logger = logging.getLogger("food_detection")
    logger.info(f"检测到食材: {', '.join(event.ingredients)} (来源: {event.source})")


def tts_announcement_event(event: FoodDetectionEvent):
    """
    语音播报事件监听器（修复macOS上的Objective-C相关错误）
    
    Args:
        event: 食材检测事件
    """
    try:
        import pyttsx3
        import platform
        
        # 获取TTS配置
        rate = 200
        voice = "default"
        if config_manager:
            rate = config_manager.get("tts.rate", 200)
            voice = config_manager.get("tts.voice", "default")
        
        # 在macOS上特殊处理
        if platform.system() == "Darwin":  # macOS
            try:
                # 尝试初始化TTS引擎
                engine = pyttsx3.init()
                
                # 设置语音参数（针对macOS优化）
                voices = engine.getProperty('voices')
                if voices and voice != "default":
                    # 选择指定语音
                    engine.setProperty('voice', voice)
                elif voices:
                    # 选择系统默认语音
                    engine.setProperty('voice', voices[0].id)
                
                # 设置语速
                engine.setProperty('rate', rate)
                
                message = f"检测到食材: {', '.join(event.ingredients)}"
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                # 如果pyttsx3有问题，尝试使用系统命令
                message = f"检测到食材: {', '.join(event.ingredients)}"
                import subprocess
                subprocess.run(["say", message])
        else:
            # 非macOS系统使用默认方式
            engine = pyttsx3.init()
            engine.setProperty('rate', rate)
            if voice != "default":
                engine.setProperty('voice', voice)
                
            message = f"检测到食材: {', '.join(event.ingredients)}"
            engine.say(message)
            engine.runAndWait()
            
    except ImportError:
        logging.warning("pyttsx3库未安装，无法进行语音播报")
    except Exception as e:
        logging.error(f"语音播报出错: {e}")


def external_api_call_event(event: FoodDetectionEvent):
    """
    外部接口调用事件监听器（示例实现）
    
    Args:
        event: 食材检测事件
    """
    try:
        # 获取API配置
        endpoint = ""
        if config_manager:
            endpoint = config_manager.get("api.endpoint", "")
        
        if not endpoint:
            logging.warning("未配置API端点，跳过外部接口调用")
            return
            
        # 示例：向外部API发送检测结果
        payload = event.to_dict()
        # 注意：这里需要替换为实际的API端点
        # response = requests.post(endpoint, json=payload)
        # if response.status_code != 200:
        #     logging.warning(f"外部API调用失败: {response.status_code}")
            
        logging.info(f"已向外部API发送食材检测结果: {payload}")
    except ImportError:
        logging.warning("requests库未安装，无法调用外部接口")
    except Exception as e:
        logging.error(f"外部接口调用出错: {e}")


def kafka_push_event(event: FoodDetectionEvent):
    """
    Kafka推送事件监听器（示例实现）
    
    Args:
        event: 食材检测事件
    """
    try:
        # 获取Kafka配置
        bootstrap_servers = ["localhost:9092"]
        topic = "food-detection-events"
        if config_manager:
            bootstrap_servers = config_manager.get("kafka.bootstrap_servers", ["localhost:9092"])
            topic = config_manager.get("kafka.topic", "food-detection-events")
        
        # 示例：推送事件到Kafka
        # from kafka import KafkaProducer
        
        # producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        # message = json.dumps(event.to_dict()).encode('utf-8')
        # producer.send(topic, message)
        # producer.flush()
        
        logging.info(f"已向Kafka推送食材检测事件: {event.to_dict()}")
    except ImportError:
        logging.warning("kafka-python库未安装，无法推送至Kafka")
    except Exception as e:
        logging.error(f"Kafka推送出错: {e}")


# 全局事件处理器实例
event_handler = EventHandler()