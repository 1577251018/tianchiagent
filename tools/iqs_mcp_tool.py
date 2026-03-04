# -*- coding: utf-8 -*-
"""
IQS MCP 搜索工具 - Multi-Agent QA System
使用原生 HTTP 调用阿里云 IQS MCP Server 进行网页搜索
"""

import httpx
import json
import re
from typing import List, Dict, Optional
import os
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from dotenv import load_dotenv
load_dotenv()
from tools.base_tool import BaseTool


class IQSSearchTool(BaseTool):
    """
    阿里云 IQS MCP 搜索工具
    通过原生 HTTP 调用 IQS MCP Server 获取网页搜索结果
    """

    def __init__(self):
        super().__init__("IQSSearch")
        self.mcp_url = os.getenv("IQS_MCP_SSE_URL", "")
        self.api_key = os.getenv("IQS_API_KEY", "")
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        self.timeout = httpx.Timeout(60.0, read=120.0)

    def execute(self, query: str, count: int = None) -> List[Dict]:
        """
        执行搜索查询

        Args:
            query: 搜索关键词
            count: 返回结果数量（IQS MCP 自行管理返回数量）

        Returns:
            搜索结果列表，每个结果包含 title, url, snippet
        """

        results = self._search_sync(query)

        return results

    def _search_sync(self, query: str) -> List[Dict]:
        """
        同步执行 MCP 搜索（使用 streamable-http 模式）
        """
        results = []

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
                    return results

                # 解析初始化响应
                init_result = init_response.json()
                session_id = init_response.headers.get("mcp-session-id")

                if session_id:
                    self.headers["mcp-session-id"] = session_id


                # 步骤2: 调用 tools/call 接口
                tool_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "common_search",
                        "arguments": {
                            "query": query
                        }
                    }
                }

                tool_response = client.post(
                    self.mcp_url,
                    content=json.dumps(tool_request).encode() + b'\n',
                    headers=self.headers
                )

                if tool_response.status_code != 200:
                    return results

                # 解析搜索结果
                tool_result = tool_response.json()

                if "result" in tool_result and "content" in tool_result["result"]:
                    for content_block in tool_result["result"]["content"]:
                        if content_block.get("type") == "text":
                            text = content_block.get("text", "")
                            parsed = self._parse_markdown_results(text)
                            results.extend(parsed)

        except Exception as e:
            print("检索出错")
        return results

    def _parse_markdown_results(self, markdown_text: str) -> List[Dict]:
        """
        解析 IQS 返回的 Markdown 格式搜索结果
        """
        results = []

        # 按段落分割（以 ## 开头的标题为分隔）
        sections = re.split(r'\n(?=##\s)', markdown_text)

        for section in sections:
            if not section.strip():
                continue

            result = {"title": "", "url": "", "snippet": ""}

            # 提取标题
            title_match = re.search(r'^##\s*(?:标题\s*)?(.*?)(?:\n|$)', section)
            if title_match:
                result["title"] = title_match.group(1).strip()

            # 提取 URL
            url_match = re.search(r'\*\*url\*\*:\s*(https?://[^\s\n]+)', section, re.IGNORECASE)
            if url_match:
                result["url"] = url_match.group(1).strip()

            # 提取 snippet
            snippet_match = re.search(r'\*\*snippet\*\*:\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
            if snippet_match:
                result["snippet"] = snippet_match.group(1).strip()[:300]

            if result["url"] or result["title"]:
                results.append(result)

        # 尝试解析标准 Markdown 链接格式
        if not results:
            link_pattern = r'\[([^\]]+)\]\((https?://[^)]+)\)'
            for match in re.finditer(link_pattern, markdown_text):
                results.append({
                    "title": match.group(1).strip(),
                    "url": match.group(2).strip(),
                    "snippet": ""
                })

        if not results and markdown_text.strip():
            results.append({
                "title": "搜索结果",
                "url": "",
                "snippet": markdown_text[:500]
            })

        return results

    def search_multiple(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """
        执行多个搜索查询
        """
        results = {}
        for query in queries:
            try:
                results[query] = self.run(query)
            except Exception as e:
                results[query] = []
        return results


# 便捷函数
def search(query: str, count: int = None) -> List[Dict]:
    """便捷搜索函数"""
    tool = IQSSearchTool()
    return tool.run(query, count)

