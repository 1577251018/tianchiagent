import os
import re
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Dict, Any

from tools.iqs_readpage_tool import IQSReadPageTool
from tools.iqs_mcp_tool import IQSSearchTool


class search_tool():
    def __init__(self):
        self.tool = IQSSearchTool()
        self.scraper_tool = IQSReadPageTool()
        self.enable_second_hop = os.getenv("SECOND_HOP_ENABLED", "1") != "0"
        self.second_hop_max_per_query = int(os.getenv("SECOND_HOP_MAX_PER_QUERY", "4"))
        self.second_hop_max_per_page = int(os.getenv("SECOND_HOP_MAX_PER_PAGE", "2"))
        self.second_hop_min_score = float(os.getenv("SECOND_HOP_MIN_SCORE", "1.0"))
        self.last_debug: Dict[str, Any] = {}

    def search(self, query: str, num_results: int = 7, context_query: str = ""):
        results = []
        first_hop_results = []
        seen_urls = set()
        debug: Dict[str, Any] = {
            "query": query,
            "context_query": context_query,
            "num_results": num_results,
            "search_api_result_count": 0,
            "first_hop_candidates": [],
            "first_hop_selected_urls": [],
            "first_hop_fetches": [],
            "second_hop_enabled": self.enable_second_hop,
            "second_hop_triggered_pages": [],
            "second_hop_candidates": [],
            "second_hop_selected": [],
            "errors": [],
            "final_result_count": 0,
        }
        try:
            temp = self.tool.run(query)
            debug["search_api_result_count"] = len(temp)
            urls_to_scrape = self._select_top_urls(temp, max_urls=num_results)
            debug["first_hop_candidates"] = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in temp[:20]
            ]
            debug["first_hop_selected_urls"] = list(urls_to_scrape)

            for url in urls_to_scrape:
                content = self.scraper_tool.run(url)
                debug["first_hop_fetches"].append(
                    {
                        "url": url,
                        "content_len": len(content) if content else 0,
                        "ok": bool(content),
                        "content_preview": (content or "")[:600],
                    }
                )
                if content:
                    for r in temp:
                        if r.get("url") == url:
                            enriched = dict(r)
                            enriched["content"] = content
                            enriched["search keyword"] = query
                            enriched["hop"] = 1
                            results.append(enriched)
                            first_hop_results.append(enriched)
                            seen_urls.add(url)
                            break

            if self.enable_second_hop and first_hop_results:
                second_hop_results, second_hop_debug = self._collect_second_hop_results(
                    first_hop_results=first_hop_results,
                    query=query,
                    context_query=context_query,
                    seen_urls=seen_urls,
                )
                debug["second_hop_triggered_pages"] = second_hop_debug.get("triggered_pages", [])
                debug["second_hop_candidates"] = second_hop_debug.get("candidates", [])
                debug["second_hop_selected"] = second_hop_debug.get("selected", [])
                results.extend(second_hop_results)

        except Exception as e:
            debug["errors"].append(str(e))
            print(f"Search failed: {e}")

        debug["final_result_count"] = len(results)
        self.last_debug = debug
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

    def _collect_second_hop_results(
        self,
        first_hop_results: List[Dict[str, Any]],
        query: str,
        context_query: str,
        seen_urls: set,
    ) -> (List[Dict[str, Any]], Dict[str, Any]):
        query_terms = self._extract_query_terms(context_query or query)
        candidates: List[Dict[str, Any]] = []
        debug_payload: Dict[str, Any] = {
            "triggered_pages": [],
            "candidates": [],
            "selected": [],
        }

        for page in first_hop_results:
            content = page.get("content", "")
            should_trigger = self._should_second_hop(content)
            if should_trigger:
                debug_payload["triggered_pages"].append(
                    {
                        "url": page.get("url", ""),
                        "title": page.get("title", ""),
                    }
                )
            if not should_trigger:
                continue

            parent_url = page.get("url", "")
            page_candidates = self._extract_link_candidates(content, parent_url)

            scored = []
            for cand in page_candidates:
                score = self._score_link_candidate(cand, query_terms, parent_url)
                if score < self.second_hop_min_score:
                    continue
                cand["score"] = score
                cand["parent_url"] = parent_url
                debug_payload["candidates"].append(
                    {
                        "parent_url": parent_url,
                        "url": cand.get("url", ""),
                        "anchor": cand.get("anchor", ""),
                        "score": score,
                    }
                )
                scored.append(cand)

            scored.sort(key=lambda item: item["score"], reverse=True)
            candidates.extend(scored[: self.second_hop_max_per_page])

        candidates.sort(key=lambda item: item["score"], reverse=True)

        second_hop_results: List[Dict[str, Any]] = []
        for cand in candidates:
            if len(second_hop_results) >= self.second_hop_max_per_query:
                break

            target_url = cand.get("url", "")
            if not target_url or target_url in seen_urls:
                continue

            content = self.scraper_tool.run(target_url)
            if not content:
                continue

            seen_urls.add(target_url)
            second_hop_results.append(
                {
                    "title": cand.get("anchor") or "Second-hop page",
                    "url": target_url,
                    "snippet": f"from: {cand.get('parent_url', '')}; anchor: {cand.get('anchor', '')}".strip("; "),
                    "content": content,
                    "search keyword": query,
                    "hop": 2,
                    "parent_url": cand.get("parent_url", ""),
                    "anchor_text": cand.get("anchor", ""),
                }
            )
            debug_payload["selected"].append(
                {
                    "url": target_url,
                    "parent_url": cand.get("parent_url", ""),
                    "anchor": cand.get("anchor", ""),
                    "content_len": len(content),
                    "content_preview": content[:600],
                }
            )

        return second_hop_results, debug_payload

    def _should_second_hop(self, content: str) -> bool:
        if not content:
            return False
        lowered = content.lower()
        trigger_terms = [
            "多义词",
            "义项",
            "展开",
            "收起",
            "正在加载",
            "javascript:;",
            "查看更多",
            "查看全部",
            "expand",
            "read more",
            "show more",
        ]
        if any(term in content or term in lowered for term in trigger_terms):
            return True

        # Link-heavy pages are often navigation stubs and need follow-up.
        markdown_link_count = len(re.findall(r"(?<!!)\[[^\]]{1,120}\]\(https?://[^)]+\)", content))
        return markdown_link_count >= 4

    def _extract_link_candidates(self, content: str, base_url: str) -> List[Dict[str, str]]:
        candidates: List[Dict[str, str]] = []
        seen = set()
        base_norm = self._normalize_url(base_url, base_url)

        for match in re.finditer(r"(?<!!)\[([^\]\n]{1,120})\]\((https?://[^)\s]+)\)", content):
            anchor = match.group(1).strip()
            target = self._normalize_url(match.group(2).strip(), base_url)
            if not target or target == base_norm or target in seen or self._is_noise_url(target):
                continue

            start = max(0, match.start() - 120)
            end = min(len(content), match.end() + 120)
            context = content[start:end].replace("\n", " ")

            seen.add(target)
            candidates.append({"anchor": anchor, "url": target, "context": context})

        return candidates

    def _score_link_candidate(self, cand: Dict[str, str], query_terms: List[str], parent_url: str) -> float:
        anchor = cand.get("anchor", "")
        context = cand.get("context", "")
        url = cand.get("url", "")
        anchor_l = anchor.lower()
        context_l = context.lower()
        url_l = url.lower()
        score = 0.0

        if 1 < len(anchor) <= 40:
            score += 1.0

        for term in query_terms:
            if term in anchor_l:
                score += 3.0
            elif term in context_l:
                score += 1.2
            elif term in url_l:
                score += 0.8

        positive_terms = [
            "义项",
            "详情",
            "词条",
            "官方",
            "人物",
            "作品",
            "动画",
            "漫画",
            "电影",
            "biography",
            "profile",
            "official",
            "details",
        ]
        negative_terms = [
            "登录",
            "注册",
            "下载",
            "广告",
            "论坛",
            "评论",
            "举报",
            "javascript",
            "app",
            "login",
            "signup",
            "register",
        ]

        if any(term in anchor or term in anchor_l for term in positive_terms):
            score += 2.0
        if any(term in context or term in context_l for term in positive_terms):
            score += 1.0
        if any(term in anchor or term in anchor_l for term in negative_terms):
            score -= 2.5
        if any(term in context or term in context_l for term in negative_terms):
            score -= 1.0

        parent_domain = self._domain(parent_url)
        target_domain = self._domain(url)
        if parent_domain and target_domain and parent_domain == target_domain:
            score += 1.0

        path = urlparse(url).path.strip()
        if path in {"", "/"}:
            score -= 3.0

        return score

    def _extract_query_terms(self, text: str) -> List[str]:
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "from",
            "this",
            "what",
            "which",
            "who",
            "when",
            "where",
            "name",
            "english",
            "between",
            "about",
        }

        terms = []
        for token in re.findall(r"[a-zA-Z]{3,}", text.lower()):
            if token not in stop_words:
                terms.append(token)

        # Keep meaningful Chinese chunks too.
        for token in re.findall(r"[\u4e00-\u9fff]{2,8}", text):
            terms.append(token)

        uniq = []
        seen = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            uniq.append(term)
        return uniq[:20]

    def _normalize_url(self, url: str, base_url: str) -> str:
        full_url = urljoin(base_url, url)
        parsed = urlparse(full_url)
        if parsed.scheme not in {"http", "https"}:
            return ""
        # Drop fragment to reduce duplicates.
        clean = parsed._replace(fragment="")
        return urlunparse(clean)

    def _is_noise_url(self, url: str) -> bool:
        lowered = url.lower()
        if "javascript:" in lowered:
            return True
        if any(lowered.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".pdf", ".zip"]):
            return True
        noise_tokens = ["login", "signup", "register", "download", "share", "comment", "gallery/list"]
        return any(token in lowered for token in noise_tokens)

    def _domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""
