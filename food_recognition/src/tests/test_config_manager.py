"""
配置管理器测试模块

作者: zhangpeng
时间: 2025-08-28
"""

import unittest
import os
import tempfile
import json
from src.config.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时配置文件
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        config_data = {
            "event_handlers": {
                "log": True,
                "tts": True,
                "api": False,
                "kafka": False
            },
            "tts": {
                "enabled": True,
                "rate": 150,
                "voice": "test_voice"
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """测试后清理"""
        os.unlink(self.temp_config.name)
    
    def test_config_manager_init_with_file(self):
        """测试配置管理器初始化（使用配置文件）"""
        config_manager = ConfigManager(self.temp_config.name)
        self.assertIsNotNone(config_manager)
        self.assertTrue(config_manager.is_event_handler_enabled("log"))
        self.assertTrue(config_manager.is_event_handler_enabled("tts"))
        self.assertFalse(config_manager.is_event_handler_enabled("api"))
        
    def test_config_manager_init_without_file(self):
        """测试配置管理器初始化（不使用配置文件）"""
        # 使用不存在的配置文件路径
        config_manager = ConfigManager("/path/does/not/exist/config.json")
        self.assertIsNotNone(config_manager)
        # 应该使用默认配置
        self.assertTrue(config_manager.is_event_handler_enabled("log"))
        self.assertFalse(config_manager.is_event_handler_enabled("tts"))
        
    def test_get_config_value(self):
        """测试获取配置值"""
        config_manager = ConfigManager(self.temp_config.name)
        self.assertEqual(config_manager.get("tts.rate"), 150)
        self.assertEqual(config_manager.get("tts.voice"), "test_voice")
        self.assertEqual(config_manager.get("nonexistent.key", "default"), "default")
        
    def test_is_event_handler_enabled(self):
        """测试事件处理器启用检查"""
        config_manager = ConfigManager(self.temp_config.name)
        self.assertTrue(config_manager.is_event_handler_enabled("log"))
        self.assertTrue(config_manager.is_event_handler_enabled("tts"))
        self.assertFalse(config_manager.is_event_handler_enabled("api"))
        self.assertFalse(config_manager.is_event_handler_enabled("kafka"))


if __name__ == '__main__':
    unittest.main()