# -*- coding: utf-8 -*-
"""
基础工具类 - Multi-Agent QA System
定义所有工具的抽象基类和通用接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from dotenv import load_dotenv
import os
load_dotenv()


class BaseTool(ABC):
    """
    工具基类
    所有具体工具（搜索、抓取、LLM调用）都继承此类
    """
    
    def __init__(self, name: str):
        """
        初始化工具
        
        Args:
            name: 工具名称，用于日志和调试
        """
        self.name = name
        self.logger = logging.getLogger(f"Tool.{name}")
        self._call_count = 0
        self._error_count = 0
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        执行工具的具体操作
        子类必须实现此方法
        
        Returns:
            工具执行结果
        """
        pass
    
    def run(self, *args, **kwargs) -> Any:
        """
        带重试逻辑的工具执行入口
        
        Returns:
            工具执行结果
        """
        self._call_count += 1
        last_error = None
        # for attempt in range(int(os.getenv("MAX_RETRIES"))):
        for attempt in range(3):
            try:
                # self.logger.debug(f"执行 {self.name}，尝试 {attempt + 1}/{int(os.getenv("MAX_RETRIES"))}")
                self.logger.debug(f"执行 {self.name}，尝试 {attempt + 1}/{3}")
                result = self.execute(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                self._error_count += 1
                self.logger.warning(f"执行失败 (尝试 {attempt + 1}): {str(e)}")
                
                if attempt < int(os.getenv("MAX_RETRIES")) - 1:
                    # 指数退避
                    delay = int(os.getenv("RETRY_DELAY")) * (int(os.getenv("RETRY_BACKOFF")) ** attempt)
                    self.logger.info(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)
        
        # 所有重试都失败
        self.logger.error(f"所有重试都失败: {str(last_error)}")
        raise last_error
    
    def get_stats(self) -> Dict[str, int]:
        """获取工具调用统计"""
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "success_rate": (self._call_count - self._error_count) / max(self._call_count, 1) * 100
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"
