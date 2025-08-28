"""
事件处理模块使用示例（可直接运行）
展示如何添加和使用各种事件监听器

作者: zhangpeng
时间: 2025-08-28
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 动态导入事件处理模块
import importlib.util
event_handler_path = os.path.join(project_root, 'src', 'events', 'event_handler.py')
spec = importlib.util.spec_from_file_location("event_handler", event_handler_path)
event_handler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event_handler_module)

# 从模块中获取需要的函数和类
event_handler = event_handler_module.event_handler
tts_announcement_event = event_handler_module.tts_announcement_event
external_api_call_event = event_handler_module.external_api_call_event
kafka_push_event = event_handler_module.kafka_push_event
FoodDetectionEvent = event_handler_module.FoodDetectionEvent

# 尝试使用配置管理器设置日志
try:
    from src.config.config_manager import config_manager
    config_manager.setup_logging()
except ImportError:
    # 如果无法导入，使用相对路径方式
    log_file = os.path.join(project_root, "logs", "food_detection.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    import logging
    from logging.handlers import RotatingFileHandler
    
    # 为food_detection记录器添加文件处理器
    logger = logging.getLogger("food_detection")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def setup_all_listeners():
    """
    设置所有事件监听器
    """
    # 添加语音播报事件监听器
    event_handler.add_listener(tts_announcement_event)
    print("已添加语音播报事件监听器")
    
    # 添加外部接口调用事件监听器
    event_handler.add_listener(external_api_call_event)
    print("已添加外部接口调用事件监听器")
    
    # 添加Kafka推送事件监听器
    event_handler.add_listener(kafka_push_event)
    print("已添加Kafka推送事件监听器")


def setup_selective_listeners():
    """
    选择性设置事件监听器
    """
    # 只添加语音播报和Kafka推送
    event_handler.add_listener(tts_announcement_event)
    event_handler.add_listener(kafka_push_event)
    print("已添加语音播报和Kafka推送事件监听器")


def test_event_trigger():
    """
    测试事件触发
    """
    print("触发测试事件...")
    event = FoodDetectionEvent(
        ingredients=["土豆", "胡萝卜", "青椒"],
        source="test"
    )
    event_handler.trigger_event(event)
    print("测试事件触发完成")
    
    # 获取日志配置并显示日志文件位置
    try:
        from src.config.config_manager import config_manager
        log_file = config_manager.get("logging.file", "logs/food_detection.log")
        print(f"日志已保存到 {log_file} 文件中")
    except ImportError:
        print("日志已保存到 logs/food_detection.log 文件中")


if __name__ == "__main__":
    print("事件处理模块使用示例")
    print("1. 添加所有事件监听器")
    print("2. 选择性添加事件监听器")
    print("3. 测试事件触发")
    print("4. 添加所有监听器并测试")
    
    choice = input("请选择操作 (1, 2, 3 或 4): ")
    
    if choice == "1":
        setup_all_listeners()
    elif choice == "2":
        setup_selective_listeners()
    elif choice == "3":
        test_event_trigger()
    elif choice == "4":
        setup_all_listeners()
        test_event_trigger()
    else:
        print("无效选择")