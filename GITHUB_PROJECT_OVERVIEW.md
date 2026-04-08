# Agent Travel

一个面向旅行规划场景的多 Agent 智能体项目，基于 FastAPI、Streamlit、LangGraph 和 OpenAI 兼容模型接口构建，支持并发分析、任务级流式输出、本地专家 skill/RAG、短期记忆、Redis 状态存储与 PostgreSQL 结果落库。

## 项目亮点

- 共享会话下的并发多 Agent 架构
- 分析型 Agent 并发执行，整合型 Agent 串行收敛
- 基于 SSE 的任务级流式输出
- 本地专家 `skill + RAG` 保留并参与最终规划
- 短期记忆可落 Redis，最终结果可落 PostgreSQL
- 支持 OpenAI 兼容模型、DuckDuckGo 搜索、MCP 天气工具

## 适用场景

这个项目适合用于：

- 旅行规划 Agent 系统原型
- 多 Agent 协作架构教学与实验
- 带状态流和事件流的智能体后端参考实现
- 结合 MCP、RAG、Redis、PostgreSQL 的完整应用示例

## 当前架构

本项目已经从早期的串行调度模式升级为：

`共享会话 + Agent 私有上下文 + 并发分析 + 串行整合 + 协调员终审`

核心流程如下：

1. 用户提交旅行请求
2. Coordinator 拆分子任务
3. `travel_advisor / weather_analyst / budget_optimizer / local_expert` 并发执行
4. Collector 汇总分析结果
5. `itinerary_planner` 生成逐日行程
6. Coordinator 做最终审阅与收口
7. Final summarizer 输出最终文档
8. 结果写入本地文件、Redis、PostgreSQL

## 架构图

```mermaid
graph TD
  U["User"] --> FE["Streamlit Frontend"]
  FE --> API["FastAPI Backend"]
  API --> TASK["Task Runtime"]

  TASK --> COOR1["Coordinator: Task Planning"]
  COOR1 --> A1["travel_advisor"]
  COOR1 --> A2["weather_analyst"]
  COOR1 --> A3["budget_optimizer"]
  COOR1 --> A4["local_expert (skill + RAG)"]

  A1 --> COL["Collector"]
  A2 --> COL
  A3 --> COL
  A4 --> COL

  COL --> ITI["itinerary_planner"]
  ITI --> COOR2["Coordinator: Final Review"]
  COOR2 --> SUM["Final Summarizer"]

  TASK --> SSE["SSE Event Stream"]
  SSE --> FE

  TASK --> REDIS["Redis: Task State / Events / Short-term Memory"]
  TASK --> PG["PostgreSQL: Final Result"]
```

## 核心模块

### 1. 多 Agent 规划引擎

后端核心位于 `backend/agents/langgraph_agents.py`，负责：

- 构建短期记忆
- 生成子任务
- 并发执行分析型 Agent
- 汇总结果
- 行程整合与协调员终审

### 2. API 与事件流

后端 API 位于 `backend/api_server.py`，负责：

- 接收规划请求
- 创建后台任务
- 输出任务状态
- 提供 SSE 流式事件接口
- 保存最终结果

### 3. 工具层

工具位于 `backend/tools/`，包括：

- DuckDuckGo 搜索工具
- MCP 天气客户端与天气服务
- 本地知识 RAG 工具
- 本地专家 skill

### 4. 存储层

存储逻辑位于 `backend/storage/persistence.py`，支持：

- Redis：任务元信息、事件流、短期记忆
- PostgreSQL：最终规划结果、Markdown、Agent 参与情况

## 短期记忆与流式输出

### 短期记忆

短期记忆是任务级状态，不做跨任务长期用户画像。当前短期记忆包含：

- `session_id`
- `shared_facts`
- `coordinator_plan`
- `collector_output`
- `itinerary_output`
- `coordinator_final_output`
- `agent_slots` 状态快照

### 流式输出

项目使用 SSE 实现任务级流式输出，事件包括：

- `task_created`
- `task_started`
- `coordinator_planned`
- `agent_started`
- `tool_called`
- `tool_completed`
- `agent_completed`
- `collector_completed`
- `itinerary_completed`
- `coordinator_finalized`
- `task_completed`

前端通过 `/stream/{task_id}` 实时消费这些事件，并在失败时回退到轮询模式。

## 技术栈

- Frontend: Streamlit
- Backend: FastAPI
- Workflow: LangGraph
- LLM: OpenAI-compatible API
- MCP: Weather MCP client/server
- RAG: Chroma-based local knowledge retrieval
- Cache / State: Redis
- Final Storage: PostgreSQL

## 启动方式

### 环境建议

- Python 3.10+
- Anaconda 环境：`agent-travel`

### 安装依赖

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 常用环境变量

```bash
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_openai_compatible_base_url
OPENAI_MODEL=your_model_name

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=postgres

QWEATHER_API_KEY=your_qweather_key
```

### 启动后端

```bash
cd backend
python api_server.py
```

### 启动前端

```bash
cd frontend
streamlit run streamlit_app.py
```

## 项目结构

```text
Agent-travel/
├─ backend/
│  ├─ agents/
│  ├─ tools/
│  ├─ storage/
│  ├─ skills/
│  ├─ config/
│  └─ api_server.py
├─ frontend/
│  └─ streamlit_app.py
├─ knowledge-rag/
├─ docs/
├─ start_backend.ps1
├─ start_frontend.ps1
└─ GITHUB_PROJECT_OVERVIEW.md
```

## 当前特色能力

- 保留本地专家 `skill` 和 `RAG` 设计
- 天气模块支持 MCP 调用与 HTTP 兜底
- 事件流支持前端实时可视化
- 已支持 Redis + PostgreSQL 接入
- 结果不仅保存文件，也可保存结构化数据库记录

## 后续可扩展方向

- 将短期记忆改为关键阶段实时 checkpoint 到 Redis
- 将 SSE 事件源从进程内内存迁移到 Redis Stream
- 支持多实例部署下的统一任务状态管理
- 增加自动化测试和回归验证
- 增加面向产品化的用户会话管理能力

## License

