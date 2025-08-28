"""
食材识别项目自定义异常类

作者: zhangpeng
时间: 2025-08-28
"""

class FoodRecognitionException(Exception):
    """食材识别基础异常类"""
    pass


class ModelLoadException(FoodRecognitionException):
    """模型加载异常"""
    def __init__(self, message: str, model_path: str = None):
        super().__init__(message)
        self.model_path = model_path


class DetectionException(FoodRecognitionException):
    """检测异常"""
    def __init__(self, message: str, source: str = None):
        super().__init__(message)
        self.source = source


class ConfigException(FoodRecognitionException):
    """配置异常"""
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message)
        self.config_key = config_key


class EventException(FoodRecognitionException):
    """事件处理异常"""
    def __init__(self, message: str, event_type: str = None):
        super().__init__(message)
        self.event_type = event_type