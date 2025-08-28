"""
事件处理模块使用示例
展示如何添加和使用各种事件监听器

作者: zhangpeng
时间: 2025-08-28
"""

from .event_handler import (
    event_handler, 
    tts_announcement_event, 
    external_api_call_event, 
    kafka_push_event
)


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


if __name__ == "__main__":
    print("事件处理模块使用示例")
    print("1. 添加所有事件监听器")
    print("2. 选择性添加事件监听器")
    
    choice = input("请选择 (1 或 2): ")
    
    if choice == "1":
        setup_all_listeners()
    elif choice == "2":
        setup_selective_listeners()
    else:
        print("无效选择")