import asyncio
import threading
import concurrent.futures
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

class SyncMCPFetchTool:
    def __init__(self):
        self.server_params = StdioServerParameters(
            command="uvx",
            args=["mcp-server-fetch", "--ignore-robots-txt"],
            env=None
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._request_queue = None
        self._init_future = concurrent.futures.Future()

    def _start_loop(self):
        """在后台线程启动事件循环"""
        asyncio.set_event_loop(self._loop)
        self._request_queue = asyncio.Queue()
        # 运行主工作协程，直到它主动退出
        self._loop.run_until_complete(self._worker_task())

    async def _worker_task(self):
        """后台核心守护任务：保证所有的 async with 都在同一个 task 里完整执行"""
        try:
            async with stdio_client(self.server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # 告诉主线程：初始化完成了！
                    self._loop.call_soon_threadsafe(self._init_future.set_result, True)

                    # 进入无限循环，等待主线程发来的抓取任务
                    while True:
                        item = await self._request_queue.get()
                        if item is None:  # 收到 None 代表收到停止信号，退出循环
                            break

                        target_url, response_future = item
                        try:
                            # 执行抓取
                            result = await session.call_tool("fetch", {"url": target_url})
                            # 将结果传回给主线程
                            self._loop.call_soon_threadsafe(response_future.set_result, result)
                        except Exception as e:
                            self._loop.call_soon_threadsafe(response_future.set_exception, e)
                        finally:
                            self._request_queue.task_done()

        except Exception as e:
            if not self._init_future.done():
                self._loop.call_soon_threadsafe(self._init_future.set_exception, e)
            else:
                print(f"后台 MCP 任务异常: {e}")

    # ================= 暴露给外部的纯同步接口 =================
    
    def start(self):
        """启动并初始化服务器"""
        print("🚀 [同步模式] 正在启动并初始化本地 MCP 服务器...")
        self._thread.start()
        # 阻塞主线程，直到后台的 _init_future 被设为完成
        self._init_future.result()
        print("✅ [同步模式] 初始化成功！\n" + "-"*40)

    def fetch(self, target_url: str):
        """抓取网页"""
        print(f"🔍 [同步模式] 正在抓取: {target_url} ...")
        response_future = concurrent.futures.Future()
        # 把要抓取的 url 和用于接收结果的 future 打包扔进后台队列
        self._loop.call_soon_threadsafe(self._request_queue.put_nowait, (target_url, response_future))
        
        try:
            # 阻塞等待后台传回结果
            return response_future.result()
        except Exception as e:
            print(f"❌ 抓取失败: {e}")
            return None

    def stop(self):
        """安全关闭服务器"""
        # 往队列里扔一个 None，让后台任务自然退出 async with 代码块
        self._loop.call_soon_threadsafe(self._request_queue.put_nowait, None)
        self._thread.join()
        print("-" * 40 + "\n🛑 [同步模式] MCP 服务器已安全关闭。")