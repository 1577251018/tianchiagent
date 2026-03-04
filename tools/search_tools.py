from typing import TypedDict, List, Dict, Any, Optional
from tools.iqs_readpage_tool import IQSReadPageTool
from tools.iqs_mcp_tool import IQSSearchTool
class search_tool():
    def __init__(self):
        self.tool = IQSSearchTool()
        self.scraper_tool = IQSReadPageTool()

    def search(self, query:str, num_results:int = 7):
        results = []
        try:
            temp = self.tool.run(query)
            urls_to_scrape = self._select_top_urls(temp, max_urls=num_results)
            for url in urls_to_scrape:
                content = self.scraper_tool.run(url)
                if content:
                    for r in temp:
                        if r['url'] == url:
                            r['content'] = content
                            r['search keyword'] = query
                            results.append(r)

        except Exception as e:
            print(f"Search failed: {e}")

        return results

    def _select_top_urls(self, results: List[Dict], max_urls: int = 2) -> List[str]:
        """Select most relevant URLs"""
        priority_domains = ["wikipedia.org", "baike.baidu.com", "zhihu.com", "britannica.com"]

        scored = []
        for r in results:
            url = r.get("url", "")
            score = 0
            for domain in priority_domains:
                if domain in url.lower():
                    score += 10
                    break
            if r.get("snippet"):
                score += 3
            scored.append((url, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [url for url, _ in scored[:max_urls]]
    
    