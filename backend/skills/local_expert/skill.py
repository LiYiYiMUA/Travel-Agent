from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple


@dataclass(frozen=True)
class LocalExpertSkillSpec:
    name: str
    version: str
    description: str
    route_rule: str
    resources: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class LocalExpertSkillInput:
    destination: str
    interests: str = ""
    query: str = ""
    top_k: int = 4


@dataclass(frozen=True)
class LocalExpertSkillOutput:
    route: str
    local_advice: str
    retrieval_count: int
    source_tags: List[str]


class LocalExpertSkill:
    """
    A reusable capability package for local travel advice.
    It encapsulates:
    - routing policy
    - retrieval orchestration
    - local planning advice synthesis
    """

    SPEC = LocalExpertSkillSpec(
        name="local_expert_skill",
        version="1.0.0",
        description=(
            "Generate planning-ready localized advice. "
            "Route to RAG for priority cities and web search otherwise."
        ),
        route_rule="destination in {北京, 上海, 广州, 杭州, 深圳} -> RAG else Search",
        resources=(
            "knowledge-rag markdown corpus",
            "chroma cloud collection",
            "duckduckgo_search/ddgs web search",
            "local advice synthesis template",
        ),
    )

    def __init__(
        self,
        *,
        normalize_city: Callable[[str], str],
        rag_priority_cities: Sequence[str],
        rag_retriever: Callable[[str, str, int], Tuple[List[str], List[str], int]],
        search_retriever: Callable[[str], Tuple[List[str], List[str], int]],
        advice_builder: Callable[[str, str, List[str], List[str]], str],
        logger,
    ) -> None:
        self._normalize_city = normalize_city
        self._rag_priority_cities = set(rag_priority_cities)
        self._rag_retriever = rag_retriever
        self._search_retriever = search_retriever
        self._advice_builder = advice_builder
        self._logger = logger

    def _build_query(self, payload: LocalExpertSkillInput) -> str:
        return payload.query.strip() or (
            f"{payload.destination} {payload.interests} 本地建议 小众景点 文化礼仪 在地美食 交通避坑"
        ).strip()

    def _route(self, payload: LocalExpertSkillInput) -> str:
        norm_city = self._normalize_city(payload.destination)
        return "rag" if norm_city in self._rag_priority_cities else "search"

    def _run_search_fallback(self, query_text: str, reason: str) -> Tuple[str, List[str], List[str], int]:
        self._logger.info(f"local_expert skill 启用 Search fallback: {reason}")
        texts, source_tags, retrieval_count = self._search_retriever(query_text)
        self._logger.info(f"local_expert skill Search fallback 命中 {retrieval_count} 条")
        return "search_fallback", texts, source_tags, retrieval_count

    def run(self, payload: LocalExpertSkillInput) -> LocalExpertSkillOutput:
        route = self._route(payload)
        route_used = route
        query_text = self._build_query(payload)
        top_k = payload.top_k if payload.top_k > 0 else 4
        self._logger.info(
            "运行 local_expert skill - "
            f"destination={payload.destination}, route={route}, top_k={top_k}, query={query_text}"
        )

        texts: List[str] = []
        source_tags: List[str] = []
        retrieval_count = 0

        if route == "rag":
            try:
                texts, source_tags, retrieval_count = self._rag_retriever(payload.destination, query_text, top_k)
                self._logger.info(f"local_expert skill RAG 命中 {retrieval_count} 条")
                if retrieval_count == 0:
                    route_used, texts, source_tags, retrieval_count = self._run_search_fallback(
                        query_text,
                        f"city-scoped RAG returned 0 hits for {payload.destination}",
                    )
            except Exception as exc:
                self._logger.warning(f"local_expert skill RAG 失败，回退 Search: {exc}")
                try:
                    route_used, texts, source_tags, retrieval_count = self._run_search_fallback(
                        query_text,
                        f"RAG error for {payload.destination}: {exc}",
                    )
                except Exception as search_exc:
                    self._logger.warning(f"local_expert skill Search fallback 失败，使用结构化默认建议: {search_exc}")
                    route_used, texts, source_tags, retrieval_count = route, [], [], 0
        else:
            try:
                texts, source_tags, retrieval_count = self._search_retriever(query_text)
                self._logger.info(f"local_expert skill Search 命中 {retrieval_count} 条")
            except Exception as exc:
                self._logger.warning(f"local_expert skill Search 失败，使用结构化默认建议: {exc}")
                texts, source_tags, retrieval_count = [], [], 0

        local_advice = self._advice_builder(payload.destination, route_used, texts, source_tags)
        return LocalExpertSkillOutput(
            route=route_used,
            local_advice=local_advice,
            retrieval_count=retrieval_count,
            source_tags=source_tags,
        )
