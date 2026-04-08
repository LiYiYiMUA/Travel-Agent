# local_expert_skill

## Purpose
Reusable capability package for local travel advice generation.

## Encapsulated capability
- Routing policy:
  - destination in `{北京, 上海, 广州, 杭州, 深圳}` => RAG
  - otherwise => Web Search
- Retrieval orchestration:
  - RAG retriever via Chroma Cloud + local knowledge corpus
  - No cross-city global RAG fallback; if city-scoped RAG has no hits, fallback to Web Search with the same city query
  - Search retriever via DuckDuckGo (`duckduckgo_search` + `ddgs` fallback)
- Advice synthesis:
  - Convert retrieval evidence into planning-ready `local_advice`
  - Output sections: `小众地点 / 文化礼仪 / 本地餐饮 / 避坑建议`
  - Include source tags

## Integration contract
- Input: `destination`, `interests`, `query`, `top_k`
- Output: `route`, `local_advice`, `retrieval_count`, `source_tags`
- Caller: `local_expert` tool adapter in `backend/tools/travel_tools.py`
