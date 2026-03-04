# -*- coding: utf-8 -*-
"""
IQS MCP 网页解析工具 - Multi-Agent QA System
使用原生 HTTP 调用阿里云 IQS MCP Server 的 readpage 工具解析网页
"""

import httpx
import json
from typing import Optional, List, Dict
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from dotenv import load_dotenv
import os
load_dotenv()
from tools.base_tool import BaseTool


class IQSReadPageTool(BaseTool):
    """
    IQS MCP 网页解析工具
    通过原生 HTTP 调用 IQS MCP Server 的 readpage 工具
    """

    def __init__(self):
        super().__init__("IQSReadPage")
        self.mcp_url = "https://iqs-mcp.aliyuncs.com/mcp-servers/iqs-mcp-server-readpage"
        self.api_key = os.getenv("IQS_API_KEY", "")
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        self.timeout = httpx.Timeout(60.0, read=120.0)

    def execute(self, url: str, use_scrape: bool = False, max_length: int = None) -> Optional[str]:
        """
        解析网页内容

        Args:
            url: 目标网页 URL
            use_scrape: 是否使用 headless browser 模式（默认 False）
            max_length: 最大内容长度

        Returns:
            网页内容（markdown 格式）
        """
        if max_length is None:
            # max_length = int(os.getenv("MAX_CONTENT_LENGTH", ""))
            max_length = 10000

        tool_name = "readpage_scrape" if use_scrape else "readpage_basic"
        self.logger.info(f"解析网页 [{tool_name}]: {url}")

        content = self._readpage_sync(url, tool_name)

        if content and (len(content) > max_length):
            content = content[:max_length] + "..."

        self.logger.info(f"提取到 {len(content) if content else 0} 字符")
        return content

    def _readpage_sync(self, url: str, tool_name: str) -> Optional[str]:
        """
        同步执行 MCP readpage
        """
        try:
            # 步骤1: 初始化会话
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "iqs-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }

            with httpx.Client(timeout=self.timeout) as client:
                # 发送初始化请求
                init_response = client.post(
                    self.mcp_url,
                    content=json.dumps(init_request).encode() + b'\n',
                    headers=self.headers
                )

                if init_response.status_code != 200:
                    self.logger.error(f"初始化失败: {init_response.status_code}")
                    # 尝试 fallback 到另一个工具
                    if tool_name == "readpage_basic":
                        return self._readpage_sync(url, "readpage_scrape")
                    return None

                # 获取 session_id
                session_id = init_response.headers.get("mcp-session-id")
                headers = dict(self.headers)
                if session_id:
                    headers["mcp-session-id"] = session_id

                self.logger.info(f"MCP readpage会话初始化成功")

                # 步骤2: 调用 tools/call 接口
                tool_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": {
                            "url": url
                        }
                    }
                }

                tool_response = client.post(
                    self.mcp_url,
                    content=json.dumps(tool_request).encode() + b'\n',
                    headers=headers
                )

                if tool_response.status_code != 200:
                    self.logger.error(f"读取页面失败: {tool_response.status_code}")
                    # 尝试 fallback
                    if tool_name == "readpage_basic":
                        return self._readpage_sync(url, "readpage_scrape")
                    return None

                # 解析结果
                tool_result = tool_response.json()

                if "result" in tool_result and "content" in tool_result["result"]:
                    for content_block in tool_result["result"]["content"]:
                        if content_block.get("type") == "text":
                            return content_block.get("text", "")

            return None

        except Exception as e:
            self.logger.error(f"MCP readpage 调用失败: {str(e)}")
            # 尝试 fallback
            if tool_name == "readpage_basic":
                return self._readpage_sync(url, "readpage_scrape")
            return None

    def scrape_multiple(self, urls: List[str], use_scrape: bool = False) -> Dict[str, Optional[str]]:
        """
        解析多个 URL
        """
        results = {}
        for url in urls:
            try:
                results[url] = self.run(url, use_scrape=use_scrape)
            except Exception as e:
                self.logger.error(f"解析失败 '{url}': {str(e)}")
                results[url] = None
        return results


# 便捷函数
def readpage(url: str, use_scrape: bool = False) -> Optional[str]:
    """便捷解析函数"""
    tool = IQSReadPageTool()
    return tool.run(url, use_scrape=use_scrape)


# if __name__ == "__main__":
#     import config
#     config.print_config_status()

#     print("\n" + "=" * 50)
#     print("IQSReadPageTool 测试")
#     print("=" * 50)

#     test_url = "https://www.example.com"

#     if config.IQS_API_KEY and config.IQS_API_KEY != "YOUR_IQS_API_KEY_HERE":
#         print(f"\n测试 URL: {test_url}")

#         # 测试 basic 模式
#         print("\n1. 测试 readpage_basic:")
#         content = readpage(test_url, use_scrape=False)
#         if content:
#             print(f"内容长度: {len(content)}")
#             print(content[:500])
#         else:
#             print("解析失败")

#         # 测试 scrape 模式
#         print("\n2. 测试 readpage_scrape:")
#         content = readpage(test_url, use_scrape=True)
#         if content:
#             print(f"内容长度: {len(content)}")
#             print(content[:500])
#         else:
#             print("解析失败")
#     else:
#         print("请先设置 IQS_API_KEY")
