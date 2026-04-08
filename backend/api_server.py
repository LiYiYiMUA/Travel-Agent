#!/usr/bin/env python3
"""
AI旅行规划智能体 - FastAPI后端服务

这个模块提供RESTful API接口，将AI旅行规划智能体包装为Web服务。
支持异步处理和实时状态更新。

主要功能：
1. 接收前端的旅行规划请求
2. 调用AI旅行规划智能体
3. 返回规划结果和状态更新
4. 提供文件下载服务
"""

import sys
import os
import asyncio
import json
import uuid
import re
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.langgraph_agents import LangGraphTravelAgents
from agents.simple_travel_agent import SimpleTravelAgent, MockTravelAgent
from config.langgraph_config import langgraph_config as config
from storage.persistence import PostgresResultStore, RedisStateStore

# 固定项目目录，避免受启动工作目录影响
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = BACKEND_DIR.parent
JSON_RESULTS_DIR = BACKEND_DIR / "results"
MARKDOWN_RESULTS_DIR = PROJECT_ROOT_DIR / "results"

# --------------------------- 日志配置 ---------------------------
def setup_api_logger():
    logger = logging.getLogger('api_server')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "backend.log", encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

api_logger = setup_api_logger()
REDIS_STATE_STORE = RedisStateStore(api_logger)
POSTGRES_RESULT_STORE = PostgresResultStore(api_logger)


def _safe_filename_component(value: str, default: str = "unknown") -> str:
    """
    将任意字符串转换为可用于 Windows 文件名的安全片段。
    """
    text = (value or default).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = text.rstrip(" .")
    return text or default


def _extract_final_plan_markdown(result: Dict[str, Any]) -> str:
    """从规划结果中提取最终 Markdown 文本。"""
    if not isinstance(result, dict):
        return "未生成可用的最终规划文本。"

    travel_plan = result.get("travel_plan", {}) if isinstance(result.get("travel_plan"), dict) else {}
    agent_outputs = result.get("agent_outputs", {}) if isinstance(result.get("agent_outputs"), dict) else {}

    # 1) 优先最终成稿字段
    primary_candidates = [
        travel_plan.get("final_plan"),
        result.get("final_plan"),
    ]
    for item in primary_candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()

    # 2) 若无成稿，优先行程规划师输出（通常最完整）
    itinerary_output = agent_outputs.get("itinerary_planner", {})
    itinerary_text = itinerary_output.get("response") if isinstance(itinerary_output, dict) else ""
    if isinstance(itinerary_text, str) and itinerary_text.strip() and "NEED_SEARCH:" not in itinerary_text:
        return itinerary_text.strip()

    # 3) 其次选择最长的已完成智能体输出
    completed_texts: List[str] = []
    for output in agent_outputs.values():
        if not isinstance(output, dict):
            continue
        status = str(output.get("status", "")).lower()
        text = output.get("response")
        if status == "completed" and isinstance(text, str) and text.strip() and "NEED_SEARCH:" not in text:
            completed_texts.append(text.strip())
    if completed_texts:
        completed_texts.sort(key=len, reverse=True)
        return completed_texts[0]

    # 4) 最后才使用 summary（避免写入过短占位文案）
    summary_candidates = [travel_plan.get("summary"), result.get("summary")]
    for summary in summary_candidates:
        if isinstance(summary, str):
            cleaned = summary.strip()
            if cleaned and len(cleaned) >= 40:
                return cleaned

    return "未生成可用的最终规划文本。"


def _build_final_markdown_report(task_id: str, result: Dict[str, Any], request: Dict[str, Any]) -> str:
    """构建最终 Markdown 报告。"""
    destination = str(request.get("destination", "未知目的地"))
    duration = request.get("duration", "未知")
    group_size = request.get("group_size", "未知")
    budget_range = request.get("budget_range", "未知")
    interests = request.get("interests", [])
    travel_dates = request.get("travel_dates", "未指定")
    if isinstance(interests, list):
        interests_text = "、".join(map(str, interests)) if interests else "无"
    else:
        interests_text = str(interests)

    final_plan = _extract_final_plan_markdown(result)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"# {destination}旅行规划指南\n\n"
        f"- 任务ID: `{task_id}`\n"
        f"- 生成时间: {now_text}\n"
        f"- 出行天数: {duration} 天\n"
        f"- 团队人数: {group_size} 人\n"
        f"- 预算范围: {budget_range}\n"
        f"- 兴趣偏好: {interests_text}\n"
        f"- 行程日期: {travel_dates}\n\n"
        f"---\n\n"
        f"{final_plan}\n"
    )


EXPECTED_AGENT_ORDER: List[str] = [
    "travel_advisor",
    "weather_analyst",
    "budget_optimizer",
    "local_expert",
    "itinerary_planner",
]


def _analyze_agent_participation(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze whether expected agents actually participated and completed.
    """
    outputs = result.get("agent_outputs", {}) if isinstance(result, dict) else {}
    if not isinstance(outputs, dict):
        outputs = {}

    participated: List[str] = []
    completed: List[str] = []
    failed: List[str] = []
    not_participating: List[str] = []

    for agent in EXPECTED_AGENT_ORDER:
        value = outputs.get(agent)
        if value is None:
            not_participating.append(agent)
            continue

        participated.append(agent)
        if isinstance(value, dict):
            status = str(value.get("status", "")).strip().lower()
            if status == "completed":
                completed.append(agent)
            elif status in {"failed", "error", "timeout"}:
                failed.append(agent)
            elif not status:
                failed.append(agent)
        elif isinstance(value, str):
            if value.strip():
                completed.append(agent)
            else:
                failed.append(agent)
        else:
            failed.append(agent)

    extra_agents = sorted([name for name in outputs.keys() if name not in EXPECTED_AGENT_ORDER])

    return {
        "expected_agents": EXPECTED_AGENT_ORDER,
        "participated_agents": sorted(participated),
        "completed_agents": sorted(completed),
        "failed_agents": sorted(failed),
        "not_participating_agents": sorted(not_participating),
        "extra_agents": extra_agents,
        "all_expected_participated": len(not_participating) == 0,
    }

# --------------------------- 应用初始化与全局配置 ---------------------------
# 创建FastAPI应用，定义对外暴露的基础信息（标题、描述、版本等）
app = FastAPI(
    title="旅小智 - AI旅行规划智能体API",
    description="🤖 旅小智：您的智能旅行规划助手 ",
    version="2.0.0"
)

# 添加CORS中间件，允许任意来源的前端访问；生产环境建议根据域名白名单收紧策略
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局字典，用于缓存当前所有规划任务的实时状态
planning_tasks: Dict[str, Dict[str, Any]] = {}

# 任务状态持久化文件路径，重启服务后可恢复未完成/历史任务状态
TASKS_FILE = "tasks_state.json"

# --------------------------- 任务状态持久化工具函数 ---------------------------
def _build_persistable_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Build file-persistable snapshots without volatile event list fields.
    """
    persistable: Dict[str, Dict[str, Any]] = {}
    for task_id, task in planning_tasks.items():
        task_copy = dict(task)
        task_copy.pop("events", None)
        task_copy.pop("next_event_seq", None)
        persistable[task_id] = task_copy
    return persistable


def _sync_task_state_to_redis(task_id: str) -> None:
    """Mirror in-process task state into Redis."""
    task = planning_tasks.get(task_id)
    if not task:
        return
    task["updated_at"] = datetime.now().isoformat()
    REDIS_STATE_STORE.upsert_task(task_id, task)


def append_task_event(
    task_id: str,
    event_type: str,
    message: str,
    *,
    progress: Optional[int] = None,
    agent: Optional[str] = None,
    status: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a state event for SSE and Redis stream.
    """
    task = planning_tasks.get(task_id)
    if not task:
        return

    next_seq = int(task.get("next_event_seq", 1))
    event_payload: Dict[str, Any] = {
        "seq": next_seq,
        "type": event_type,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    if progress is not None:
        event_payload["progress"] = int(progress)
    if agent:
        event_payload["agent"] = agent
    if status:
        event_payload["status"] = status
    if data is not None:
        event_payload["data"] = data

    events = task.setdefault("events", [])
    events.append(event_payload)
    if len(events) > 1000:
        task["events"] = events[-1000:]
    task["next_event_seq"] = next_seq + 1

    REDIS_STATE_STORE.append_event(task_id, event_payload)
    _sync_task_state_to_redis(task_id)


def save_tasks_state():
    """保存任务状态到文件"""
    try:
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(_build_persistable_tasks(), f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        api_logger.error(f"保存任务状态失败: {e}")


def load_tasks_state():
    """从文件加载任务状态"""
    global planning_tasks
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                planning_tasks = json.load(f)
            for task in planning_tasks.values():
                task.setdefault("events", [])
                task.setdefault("next_event_seq", 1)
            for task_id in list(planning_tasks.keys()):
                _sync_task_state_to_redis(task_id)
            api_logger.info(f"已加载 {len(planning_tasks)} 个任务状态")
        else:
            api_logger.info("任务状态文件不存在，使用空状态")
    except Exception as e:
        api_logger.error(f"加载任务状态失败: {e}")
        planning_tasks = {}


# 启动时加载任务状态
load_tasks_state()

# --------------------------- 数据模型定义 ---------------------------
class TravelRequest(BaseModel):
    """
    旅行规划请求模型

    用于接受客户端/前端提交的旅行规划需求。该模型定义了用户提交给智能体系统的所有关键信息参数，
    包含从基础出行时间、目的地，到细致偏好（如兴趣、饮食禁忌、预算、交通与住宿等），
    以全面支撑多智能体的任务分工与细致化规划。

    字段说明：
        - destination (str): 旅行目的地，例如“杭州”。
        - start_date (str): 旅行开始日期，格式如“2025-08-14”。
        - end_date (str): 旅行结束日期，格式如“2025-08-17”。
        - budget_range (str): 期望的预算区间或类型，例如“经济型 (300-800元/天)”。
        - group_size (int): 出行人数。
        - interests (list[str]): 兴趣偏好列表，如 ["美食","徒步"]。
        - dietary_restrictions (str): 饮食禁忌或特殊偏好（如“全素”），默认为空字符串。
        - activity_level (str): 活动强度（如“适中”、“轻松”、“高强度”），默认“适中”。
        - travel_style (str): 旅行风格（如“探索者”、“休闲者”），默认“探索者”。
        - transportation_preference (str): 交通方式偏好（如“自驾”、“公共交通”），默认“公共交通”。
        - accommodation_preference (str): 住宿方式偏好（如“酒店”、“民宿”），默认“酒店”。
        - special_occasion (str): 是否有特殊场合（如“生日”、“纪念日”），没有则为空字符串。
        - special_requirements (str): 其他特殊需求（如“无障碍房间”），没有则为空字符串。
        - currency (str): 预算币种，默认“CNY”。

    注意事项：
        此模型作为前端与后端/智能体主控交互的数据标准，在任务派发、多智能体决策、状态持久化等多个核心模块中反复使用。
    """
    destination: str  # 目的地（如“杭州”）
    start_date: str  # 出发日期，格式如“2025-08-14”
    end_date: str  # 返回日期，格式如“2025-08-17”
    budget_range: str  # 预算范围（如“经济型 (300-800元/天)”）
    group_size: int  # 出行人数
    interests: list[str] = []  # 兴趣偏好，如["美食","徒步"]
    dietary_restrictions: str = ""  # 饮食禁忌或偏好，默认为空
    activity_level: str = "适中"  # 活动强度（如“适中”、“轻松”或“高强度”）
    travel_style: str = "探索者"  # 旅行风格（如“探索者”、“休闲者”）
    transportation_preference: str = "公共交通"  # 交通偏好，如“自驾”、“公共交通”
    accommodation_preference: str = "酒店"  # 住宿偏好，如“酒店”、“民宿”
    special_occasion: str = ""  # 特殊场合（如“生日”、“纪念日”），没有则为空
    special_requirements: str = ""  # 其他特殊需求，如“无障碍房间”，没有则为空
    currency: str = "CNY"  # 预算币种，默认为人民币（CNY）

class PlanningResponse(BaseModel):
    """规划响应模型"""
    task_id: str
    status: str
    message: str

class PlanningStatus(BaseModel):
    """规划状态模型"""
    task_id: str
    status: str
    progress: int
    current_agent: str
    message: str
    result: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """自然语言交互请求模型"""
    message: str  # 用户的自然语言输入
    
class ChatResponse(BaseModel):
    """自然语言交互响应模型"""
    understood: bool  # 是否理解用户意图
    extracted_info: Dict[str, Any]  # 提取的旅行信息
    missing_info: list[str]  # 缺失的信息
    clarification: str  # 需要澄清的问题
    can_proceed: bool  # 是否可以直接创建规划任务
    task_id: Optional[str] = None  # 如果可以直接创建，返回任务ID

# --------------------------- 路由定义 ---------------------------
@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "name": "旅小智",
        "slogan": "您的智能旅行规划助手",
        "message": "🤖 旅小智 - AI旅行规划智能体API",
        "version": "2.0.0",
        "status": "运行中",
        "features": [
            "💬 自然语言交互",
            "🤖 多智能体协作",
            "🎯 个性化规划",
            "⚡ 实时响应"
        ],
        "agents": [
            "🎯 协调员智能体",
            "✈️ 旅行顾问",
            "💰 预算优化师", 
            "🌤️ 天气分析师",
            "🏠 当地专家",
            "📅 行程规划师"
        ],
        "endpoints": {
            "chat": "/chat - 自然语言交互",
            "plan": "/plan - 创建旅行规划",
            "status": "/status/{task_id} - 查询任务状态",
            "download": "/download/{task_id} - 下载结果",
            "docs": "/docs - API文档"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查 OpenAI 兼容 API 密钥
        if not config.OPENAI_API_KEY:
            return {
                "status": "warning", 
                "message": "OPENAI_API_KEY 未配置",
                "llm_model": config.OPENAI_MODEL,
                "api_key_configured": False,
                "timestamp": datetime.now().isoformat()
            }
        
        # 检查系统资源
        import psutil
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "status": "healthy",
            "llm_model": config.OPENAI_MODEL,
            "api_key_configured": bool(config.OPENAI_API_KEY),
            "system_info": {
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory_info.percent}%",
                "memory_available": f"{memory_info.available / 1024 / 1024 / 1024:.1f}GB"
            },
            "active_tasks": len(planning_tasks),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        api_logger.error(f"健康检查错误: {e}")
        return {
            "status": "error", 
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# --------------------------- 异步执行核心任务 ---------------------------
async def _legacy_run_planning_task(task_id: str, travel_request: Dict[str, Any]):
    """
    异步执行旅行规划任务

    后台协程负责整个 LangGraph 多智能体推理流程，核心步骤如下：
        1. 更新任务状态进度条，并构造 LangGraph 所需的标准化请求 `langgraph_request`；
        2. 在异步上下文中调用 `LangGraphTravelAgents`，通过线程池避免阻塞事件循环；
        3. 设定超时与异常回退策略：若 LangGraph 超时或执行失败，则自动降级至 SimpleTravelAgent；
        4. 规划成功后保存结果、写入文件；若失败或异常，则返回简化方案并记录错误信息。

    该函数不会阻塞 API 响应，由 `BackgroundTasks` 在后台运行，确保接口响应迅速。
    """
    try:
        api_logger.info(f"开始执行任务 {task_id} | 请求: {json.dumps(travel_request, ensure_ascii=False)}")
        
        # 更新任务状态
        planning_tasks[task_id]["status"] = "processing"
        planning_tasks[task_id]["progress"] = 10
        planning_tasks[task_id]["message"] = "正在初始化AI旅行规划智能体..."
        
        # 模拟处理时间，避免立即完成
        await asyncio.sleep(1)
        
        planning_tasks[task_id]["progress"] = 30
        planning_tasks[task_id]["message"] = "多智能体系统已启动，开始协作规划..."
        
        await asyncio.sleep(1)
        
        # 转换请求格式
        langgraph_request = {
            "destination": travel_request["destination"],
            "duration": travel_request.get("duration", 7),
            "budget_range": travel_request["budget_range"],
            "interests": travel_request["interests"],
            "group_size": travel_request["group_size"],
            "travel_dates": f"{travel_request['start_date']} 至 {travel_request['end_date']}",
            "transportation_preference": travel_request.get("transportation_preference", "公共交通"),
            "accommodation_preference": travel_request.get("accommodation_preference", "酒店")
        }
        
        planning_tasks[task_id]["progress"] = 50
        planning_tasks[task_id]["message"] = "智能体团队正在协作分析..."
        
        await asyncio.sleep(1)
        
        api_logger.info(f"任务 {task_id}: 开始LangGraph处理")
        
        try:
            # 使用asyncio.wait_for添加超时控制
            async def run_langgraph():
                """封装 LangGraph 智能体执行流程，便于统一超时处理"""
                # 初始化AI旅行规划智能体
                api_logger.info(f"任务 {task_id}: 初始化AI旅行规划智能体")
                planning_tasks[task_id]["progress"] = 50
                planning_tasks[task_id]["message"] = "初始化AI旅行规划智能体..."

                try:
                    travel_agents = LangGraphTravelAgents()
                    api_logger.info(f"任务 {task_id}: AI旅行规划智能体初始化完成")

                    planning_tasks[task_id]["progress"] = 60
                    planning_tasks[task_id]["message"] = "开始多智能体协作..."

                    api_logger.info(f"任务 {task_id}: 执行旅行规划")
                    # 在线程池中执行规划，避免阻塞
                    import concurrent.futures

                    def run_planning():
                        """在线程池中实际执行多智能体规划，保持事件循环顺畅"""
                        return travel_agents.run_travel_planning(langgraph_request)

                    # 使用线程池执行，设置超时
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        try:
                            loop = asyncio.get_running_loop()
                            # 等待最多8分钟，避免多智能体在网络搜索场景下被过早降级
                            result = await asyncio.wait_for(
                                loop.run_in_executor(executor, run_planning),
                                timeout=480.0
                            )
                            api_logger.info(f"任务 {task_id}: LangGraph执行完成，结果: {result.get('success', False)}")
                            return result
                        except asyncio.TimeoutError:
                            api_logger.warning(f"任务 {task_id}: LangGraph执行超时，尝试使用简化版本")
                            planning_tasks[task_id]["progress"] = 80
                            planning_tasks[task_id]["message"] = "LangGraph超时，使用简化版本..."

                            # 使用简化版本作为备选方案
                            simple_agent = SimpleTravelAgent()
                            return simple_agent.run_travel_planning(langgraph_request)

                        except Exception as e:
                            api_logger.error(f"任务 {task_id}: LangGraph执行异常: {str(e)}，尝试使用简化版本")
                            planning_tasks[task_id]["progress"] = 80
                            planning_tasks[task_id]["message"] = "LangGraph异常，使用简化版本..."

                            # 使用简化版本作为备选方案
                            simple_agent = SimpleTravelAgent()
                            return simple_agent.run_travel_planning(langgraph_request)

                except Exception as e:
                    api_logger.error(f"任务 {task_id}: 初始化LangGraph失败: {str(e)}")
                    return {
                        "success": False,
                        "error": f"初始化失败: {str(e)}",
                        "travel_plan": {},
                        "agent_outputs": {},
                        "planning_complete": False
                    }
            
            # 设置600秒超时（10分钟），覆盖多智能体+工具调用的完整链路
            result = await asyncio.wait_for(run_langgraph(), timeout=600.0)
            
            api_logger.info(f"任务 {task_id}: LangGraph处理完成")
            
            if result["success"]:
                planning_tasks[task_id]["status"] = "completed"
                planning_tasks[task_id]["progress"] = 100
                planning_tasks[task_id]["message"] = "旅行规划完成！"
                planning_tasks[task_id]["result"] = result

                # 保存任务状态
                save_tasks_state()
                
                # 保存结果到文件
                await save_planning_result(task_id, result, langgraph_request)
                
            else:
                planning_tasks[task_id]["status"] = "failed"
                planning_tasks[task_id]["message"] = f"规划失败: {result.get('error', '未知错误')}"
                
        except asyncio.TimeoutError:
            api_logger.warning(f"任务 {task_id}: LangGraph处理超时")
            # 超时处理，提供简化响应
            simplified_result = {
                "success": True,
                "travel_plan": {
                    "destination": travel_request["destination"],
                    "duration": travel_request.get("duration", 7),
                    "budget_range": travel_request["budget_range"],
                    "group_size": travel_request["group_size"],
                    "travel_dates": f"{travel_request['start_date']} 至 {travel_request['end_date']}",
                    "transportation_preference": travel_request.get("transportation_preference", "公共交通"),
                    "accommodation_preference": travel_request.get("accommodation_preference", "酒店"),
                    "summary": f"为{travel_request['destination']}制定的{travel_request.get('duration', 7)}天旅行计划（快速模式）"
                },
                "agent_outputs": {
                    "system_message": {
                        "response": f"由于系统负载较高，为您提供快速旅行计划。目的地：{travel_request['destination']}，预算：{travel_request['budget_range']}，人数：{travel_request['group_size']}人。建议您关注当地的热门景点、特色美食和文化体验。",
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed"
                    }
                },
                "total_iterations": 1,
                "planning_complete": True
            }
            
            planning_tasks[task_id]["status"] = "completed"
            planning_tasks[task_id]["progress"] = 100
            planning_tasks[task_id]["message"] = "旅行规划完成（快速模式）"
            planning_tasks[task_id]["result"] = simplified_result
            
            # 保存简化结果
            await save_planning_result(task_id, simplified_result, langgraph_request)
                
        except Exception as agent_error:
            # 如果AI旅行规划智能体出错，提供一个简化的响应
            api_logger.error(f"任务 {task_id}: AI旅行规划智能体错误: {str(agent_error)}")
            
            # 创建一个简化的旅行计划作为回退
            simplified_result = {
                "success": True,
                "travel_plan": {
                    "destination": travel_request["destination"],
                    "duration": travel_request.get("duration", 7),
                    "budget_range": travel_request["budget_range"],
                    "group_size": travel_request["group_size"],
                    "travel_dates": f"{travel_request['start_date']} 至 {travel_request['end_date']}",
                    "transportation_preference": travel_request.get("transportation_preference", "公共交通"),
                    "accommodation_preference": travel_request.get("accommodation_preference", "酒店"),
                    "summary": f"为{travel_request['destination']}制定的{travel_request.get('duration', 7)}天旅行计划"
                },
                "agent_outputs": {
                    "system_message": {
                        "response": f"系统正在维护中，为您提供基础的旅行计划框架。目的地：{travel_request['destination']}，预算：{travel_request['budget_range']}，人数：{travel_request['group_size']}人。建议提前了解当地的交通、住宿和主要景点信息。",
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed"
                    }
                },
                "total_iterations": 1,
                "planning_complete": True
            }
            
            planning_tasks[task_id]["status"] = "completed"
            planning_tasks[task_id]["progress"] = 100
            planning_tasks[task_id]["message"] = "旅行规划完成（简化模式）"
            planning_tasks[task_id]["result"] = simplified_result
            
            # 保存简化结果
            await save_planning_result(task_id, simplified_result, langgraph_request)
            
        api_logger.info(f"任务 {task_id}: 执行完成")
            
    except Exception as e:
        planning_tasks[task_id]["status"] = "failed"
        planning_tasks[task_id]["message"] = f"系统错误: {str(e)}"
        api_logger.error(f"任务 {task_id}: 规划任务执行错误: {str(e)}")
    finally:
        # 无论成功或失败都持久化，避免进程重启后状态丢失
        save_tasks_state()

# --------------------------- 规划结果输出工具函数 ---------------------------
async def run_planning_task(task_id: str, travel_request: Dict[str, Any]):
    """
    Overridden runtime task executor with event streaming and Redis memory sync.
    """
    try:
        api_logger.info(
            f"开始执行任务 {task_id} | 请求: {json.dumps(travel_request, ensure_ascii=False)}"
        )

        def update_task_state(
            *,
            status: Optional[str] = None,
            progress: Optional[int] = None,
            message: Optional[str] = None,
            agent: Optional[str] = None,
            event_type: str = "task_update",
            data: Optional[Dict[str, Any]] = None,
        ) -> None:
            task = planning_tasks.get(task_id)
            if not task:
                return
            if status is not None:
                task["status"] = status
            if progress is not None:
                task["progress"] = int(progress)
            if message is not None:
                task["message"] = message
            if agent is not None:
                task["current_agent"] = agent
            append_task_event(
                task_id,
                event_type,
                task.get("message", ""),
                progress=task.get("progress"),
                status=task.get("status"),
                agent=task.get("current_agent"),
                data=data,
            )

        update_task_state(
            status="processing",
            progress=5,
            message="任务进入执行队列，准备初始化并发规划引擎。",
            agent="system",
            event_type="task_started",
        )

        langgraph_request = {
            "destination": travel_request["destination"],
            "duration": travel_request.get("duration", 7),
            "budget_range": travel_request["budget_range"],
            "interests": travel_request["interests"],
            "group_size": travel_request["group_size"],
            "travel_dates": f"{travel_request['start_date']} 至 {travel_request['end_date']}",
            "transportation_preference": travel_request.get("transportation_preference", "公共交通"),
            "accommodation_preference": travel_request.get("accommodation_preference", "酒店"),
        }

        update_task_state(
            progress=12,
            message="请求标准化完成，准备启动并发多 Agent 执行。",
            agent="coordinator",
            event_type="request_normalized",
        )

        async def run_langgraph() -> Dict[str, Any]:
            update_task_state(
                progress=18,
                message="初始化并发规划引擎...",
                agent="coordinator",
                event_type="planner_initializing",
            )
            travel_agents = LangGraphTravelAgents()

            def planner_event_callback(event: Dict[str, Any]) -> None:
                event_type = str(event.get("type", "planner_event"))
                message = str(event.get("message", ""))
                progress_val = event.get("progress")
                status_val = event.get("status")
                agent_val = event.get("agent")
                data_val = event.get("data")
                update_task_state(
                    status=str(status_val) if status_val else None,
                    progress=int(progress_val) if progress_val is not None else None,
                    message=message if message else None,
                    agent=str(agent_val) if agent_val else None,
                    event_type=event_type,
                    data=data_val if isinstance(data_val, dict) else None,
                )

            import concurrent.futures

            def run_planning() -> Dict[str, Any]:
                return travel_agents.run_travel_planning(
                    langgraph_request,
                    event_callback=planner_event_callback,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_running_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(executor, run_planning),
                    timeout=480.0,
                )

        try:
            result = await asyncio.wait_for(run_langgraph(), timeout=600.0)
            if result.get("success"):
                participation = _analyze_agent_participation(result)
                result["agent_participation"] = participation

                merged_missing_agents = sorted(
                    set(result.get("missing_agents", []))
                    | set(participation.get("not_participating_agents", []))
                )
                result["missing_agents"] = merged_missing_agents
                if merged_missing_agents:
                    result["planning_complete"] = False
                    api_logger.warning(
                        f"任务 {task_id}: 部分智能体未参与 -> {merged_missing_agents}"
                    )
                    append_task_event(
                        task_id,
                        "agent_participation_warning",
                        "部分智能体未参与，已记录缺失清单。",
                        progress=95,
                        status="processing",
                        agent="collector",
                        data={
                            "missing_agents": merged_missing_agents,
                            "failed_agents": participation.get("failed_agents", []),
                        },
                    )

                planning_tasks[task_id]["status"] = "completed"
                planning_tasks[task_id]["progress"] = 100
                planning_tasks[task_id]["current_agent"] = "summarizer"
                planning_tasks[task_id]["message"] = "旅行规划完成！"
                planning_tasks[task_id]["result"] = result
                append_task_event(
                    task_id,
                    "task_completed",
                    "旅行规划完成！",
                    progress=100,
                    status="completed",
                    agent="summarizer",
                )
                short_term_memory = result.get("short_term_memory")
                if isinstance(short_term_memory, dict):
                    REDIS_STATE_STORE.save_short_term_memory(task_id, short_term_memory)
                await save_planning_result(task_id, result, langgraph_request)
            else:
                planning_tasks[task_id]["status"] = "failed"
                planning_tasks[task_id]["progress"] = 100
                planning_tasks[task_id]["message"] = (
                    f"规划失败: {result.get('error', '未知错误')}"
                )
                append_task_event(
                    task_id,
                    "task_failed",
                    planning_tasks[task_id]["message"],
                    progress=100,
                    status="failed",
                    agent="system",
                )
        except Exception as agent_error:
            planning_tasks[task_id]["status"] = "failed"
            planning_tasks[task_id]["progress"] = 100
            planning_tasks[task_id]["current_agent"] = "system"
            planning_tasks[task_id]["message"] = f"规划异常: {str(agent_error)}"
            append_task_event(
                task_id,
                "task_failed",
                planning_tasks[task_id]["message"],
                progress=100,
                status="failed",
                agent="system",
            )
            api_logger.error(f"任务 {task_id}: 规划执行异常: {agent_error}")
    except Exception as e:
        planning_tasks[task_id]["status"] = "failed"
        planning_tasks[task_id]["message"] = f"系统错误: {str(e)}"
        planning_tasks[task_id]["progress"] = 100
        append_task_event(
            task_id,
            "task_failed",
            planning_tasks[task_id]["message"],
            progress=100,
            status="failed",
            agent="system",
        )
        api_logger.error(f"任务 {task_id}: 规划任务执行错误: {str(e)}")
    finally:
        save_tasks_state()
        _sync_task_state_to_redis(task_id)


async def save_planning_result(task_id: str, result: Dict[str, Any], request: Dict[str, Any]):
    """
    保存规划结果到文件

    将规划请求、结果及时间戳封装为 JSON 存入固定目录 `03-agent-build-docker-deploy/results/`，
    文件命名包含目的地与时间，
    便于后续归档。该函数在完成主任务后调用，确保生成的报告可以被用户下载或复盘。
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = _safe_filename_component(str(request.get("destination", "unknown")).replace(" ", "_"))
        filename = f"旅行计划_{destination}_{timestamp}.json"
        filepath = JSON_RESULTS_DIR / filename
        
        # 确保结果目录存在（固定到项目根下的 results）
        JSON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON格式
        save_data = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "request": request,
            "result": result
        }
        
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        planning_tasks[task_id]["result_file"] = filename
        planning_tasks[task_id]["result_path"] = str(filepath)

        markdown_filename = f"{destination}-{request.get('group_size', 1)}人-旅行规划指南-{timestamp}.md"
        markdown_filepath = MARKDOWN_RESULTS_DIR / markdown_filename
        MARKDOWN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        markdown_content = _build_final_markdown_report(task_id, result, request)
        with markdown_filepath.open('w', encoding='utf-8') as f:
            f.write(markdown_content)

        planning_tasks[task_id]["result_markdown_file"] = markdown_filename
        planning_tasks[task_id]["result_markdown_path"] = str(markdown_filepath)

        participation = (
            result.get("agent_participation")
            if isinstance(result.get("agent_participation"), dict)
            else _analyze_agent_participation(result)
        )
        result["agent_participation"] = participation
        missing_agents = sorted(
            set(result.get("missing_agents", []))
            | set(participation.get("not_participating_agents", []))
        )
        result["missing_agents"] = missing_agents

        short_term_memory = result.get("short_term_memory")
        if isinstance(short_term_memory, dict):
            REDIS_STATE_STORE.save_short_term_memory(task_id, short_term_memory)

        POSTGRES_RESULT_STORE.upsert_result(
            task_id,
            request,
            result,
            status=str(planning_tasks.get(task_id, {}).get("status", "completed")),
            result_file=filename,
            result_markdown_file=markdown_filename,
            final_plan_markdown=markdown_content,
            missing_agents_override=missing_agents,
            agent_participation=participation,
        )

        save_tasks_state()
        _sync_task_state_to_redis(task_id)

        api_logger.info(f"任务 {task_id}: JSON 已保存 -> {filepath}")
        api_logger.info(f"任务 {task_id}: Markdown 已保存 -> {markdown_filepath}")
        if POSTGRES_RESULT_STORE.enabled:
            api_logger.info(f"任务 {task_id}: PostgreSQL 已完成 upsert")
        
    except Exception as e:
        api_logger.error(f"保存结果文件时出错: {str(e)}")

# --------------------------- API 路由：创建、查询、下载 ---------------------------
@app.post("/plan", response_model=PlanningResponse)
async def create_travel_plan(request: TravelRequest, background_tasks: BackgroundTasks):
    """
    创建旅行规划任务

    该接口负责接收前端提交的详细旅行需求，初始化任务状态并触发后台异步执行：
        1. 生成唯一的 task_id，作为后续查询的关键主键；
        2. 依据起止日期计算旅行天数，写入请求体供多智能体使用；
        3. 将任务存入全局状态字典 `planning_tasks`，并立即持久化到本地文件；
        4. 投递后台任务 `run_planning_task`，由事件循环异步执行，保证接口快速响应。

    请求成功后返回 `PlanningResponse`，调用方可通过 task_id 轮询 `/status/{task_id}` 获取进度。
    """
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 计算旅行天数
        from datetime import datetime
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        duration = (end_date - start_date).days + 1
        if duration <= 0:
            raise HTTPException(status_code=400, detail="结束日期必须晚于或等于开始日期")
        
        # 转换请求为字典
        travel_request = request.model_dump()
        travel_request["duration"] = duration
        
        # 初始化任务状态
        planning_tasks[task_id] = {
            "task_id": task_id,
            "status": "started",
            "progress": 0,
            "current_agent": "系统初始化",
            "message": "任务已创建，准备开始规划...",
            "created_at": datetime.now().isoformat(),
            "request": travel_request,
            "result": None,
            "events": [],
            "next_event_seq": 1,
        }

        append_task_event(
            task_id,
            "task_created",
            "任务已创建，等待执行。",
            progress=0,
            status="started",
            agent="system",
        )

        # 保存任务状态
        save_tasks_state()
        
        # 添加后台任务
        # 这里通过 FastAPI 提供的 BackgroundTasks 功能，把“旅行规划任务”的实际执行放到后台异步运行。
        # 这样做的好处是接口能够立即响应，并不会因耗时的AI推理阻塞前端用户。
        # add_task 的第一个参数是要执行的函数（run_planning_task），
        # 后面的参数（task_id, travel_request）是传递给该函数的实际参数。
        # run_planning_task 用于具体执行业务逻辑（AI旅行规划），
        # 而 background_tasks.add_task 会在响应完成后自动在后台启动它。
        background_tasks.add_task(run_planning_task, task_id, travel_request)
        
        return PlanningResponse(
            task_id=task_id,
            status="started",
            message="旅行规划任务已启动，请使用task_id查询进度"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建规划任务失败: {str(e)}")

@app.get("/status/{task_id}", response_model=PlanningStatus)
async def get_planning_status(task_id: str):
    """
    获取规划任务状态

    根据 task_id 读取内存中的任务状态，返回进度条（0-100）、当前执行智能体/阶段提示、
    文本消息以及完成后缓存的最终结果。若任务不存在则返回 404。
    """
    try:
        api_logger.info(f"状态查询: {task_id}")

        if task_id not in planning_tasks:
            api_logger.warning(f"任务不存在: {task_id}")
            raise HTTPException(status_code=404, detail="任务不存在")

        task = planning_tasks[task_id]
        api_logger.info(f"任务状态: {task['status']}, 进度: {task['progress']}%")

        return PlanningStatus(
            task_id=task_id,
            status=task["status"],
            progress=task["progress"],
            current_agent=task["current_agent"],
            message=task["message"],
            result=task["result"]
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"状态查询错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"状态查询失败: {str(e)}")

@app.get("/stream/{task_id}")
async def stream_planning_events(task_id: str):
    """
    SSE stream for task state/events.
    """
    if task_id not in planning_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    async def event_generator():
        last_seq = 0
        keepalive_interval = 1.0
        max_idle_seconds = 900
        started_at = datetime.now()

        while True:
            task = planning_tasks.get(task_id)
            if not task:
                payload = {"type": "task_missing", "message": "任务不存在", "status": "failed"}
                yield f"event: task_missing\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

            events = task.get("events", [])
            new_events = [evt for evt in events if int(evt.get("seq", 0)) > last_seq]
            if new_events:
                for event in new_events:
                    seq = int(event.get("seq", last_seq + 1))
                    last_seq = max(last_seq, seq)
                    payload = dict(event)
                    payload.setdefault("task_id", task_id)
                    yield (
                        f"id: {seq}\n"
                        f"event: {payload.get('type', 'task_event')}\n"
                        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    )
            else:
                elapsed = (datetime.now() - started_at).total_seconds()
                if elapsed > max_idle_seconds:
                    payload = {"type": "stream_timeout", "task_id": task_id}
                    yield f"event: stream_timeout\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    break
                yield ": keepalive\n\n"

            if task.get("status") in {"completed", "failed"}:
                summary_payload = {
                    "type": "task_terminal",
                    "task_id": task_id,
                    "status": task.get("status"),
                    "progress": task.get("progress"),
                    "message": task.get("message"),
                }
                yield f"event: task_terminal\ndata: {json.dumps(summary_payload, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(keepalive_interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    下载规划结果文件

    如果任务执行成功并生成结果文件，则按照 task_id 寻址项目根 `results/` 目录下的 JSON 文件，
    返回 `FileResponse` 供调用方下载。若文件不存在或任务无结果，将抛出 404。
    """
    if task_id not in planning_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = planning_tasks[task_id]
    if "result_file" not in task:
        raise HTTPException(status_code=404, detail="结果文件不存在")
    
    # 优先读取持久化的绝对路径，兼容历史任务则回退到固定结果目录
    filepath = Path(task.get("result_path", str(JSON_RESULTS_DIR / task["result_file"])))
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=str(filepath),
        filename=task["result_file"],
        media_type='application/json'
    )

# --------------------------- 辅助路由：任务列表、简化/模拟模式 ---------------------------
@app.get("/tasks")
async def list_tasks():
    """
    列出所有任务

    将当前内存中的 `planning_tasks` 转化为摘要列表，便于调试或在管理端展示所有历史任务。
    每个任务包含 task_id、状态、创建时间及目的地信息。
    """
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "created_at": task["created_at"],
                "destination": task["request"].get("destination", "未知")
            }
            for task_id, task in planning_tasks.items()
        ]
    }

@app.post("/simple-plan")
async def simple_travel_plan(request: TravelRequest, background_tasks: BackgroundTasks):
    """
    简化版旅行规划（使用简化智能体）

    使用 `SimpleTravelAgent` 同步生成旅行方案，适用于快速响应或 LangGraph 资源不足场景。
    仍然以异步后台任务方式执行，流程与完整版类似，但智能体数量更少、执行逻辑更简单。
    """
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 计算旅行天数
        from datetime import datetime
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        duration = (end_date - start_date).days + 1
        if duration <= 0:
            raise HTTPException(status_code=400, detail="结束日期必须晚于或等于开始日期")

        # 转换请求为字典
        travel_request = request.model_dump()
        travel_request["duration"] = duration

        # 初始化任务状态
        planning_tasks[task_id] = {
            "task_id": task_id,
            "status": "started",
            "progress": 0,
            "current_agent": "简化智能体",
            "message": "任务已创建，准备开始简化规划...",
            "created_at": datetime.now().isoformat(),
            "request": travel_request,
            "result": None
        }
        save_tasks_state()

        # 添加后台任务
        async def run_simple_planning():
            """运行简化智能体规划逻辑，保持与完整版相同的状态更新流程"""
            try:
                planning_tasks[task_id]["status"] = "processing"
                planning_tasks[task_id]["progress"] = 30
                planning_tasks[task_id]["message"] = "正在使用简化智能体规划..."

                simple_agent = SimpleTravelAgent()
                result = simple_agent.run_travel_planning(travel_request)

                if result["success"]:
                    planning_tasks[task_id]["status"] = "completed"
                    planning_tasks[task_id]["progress"] = 100
                    planning_tasks[task_id]["message"] = "简化规划完成！"
                    planning_tasks[task_id]["result"] = result

                    # 保存结果到文件
                    await save_planning_result(task_id, result, travel_request)
                else:
                    planning_tasks[task_id]["status"] = "failed"
                    planning_tasks[task_id]["message"] = f"简化规划失败: {result.get('error', '未知错误')}"

            except Exception as e:
                planning_tasks[task_id]["status"] = "failed"
                planning_tasks[task_id]["message"] = f"简化规划异常: {str(e)}"
            finally:
                save_tasks_state()

        background_tasks.add_task(run_simple_planning)

        return PlanningResponse(
            task_id=task_id,
            status="started",
            message="简化版旅行规划任务已启动"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建简化规划任务失败: {str(e)}")

@app.post("/mock-plan")
async def mock_travel_plan(request: TravelRequest):
    """
    模拟旅行规划（用于测试，立即返回结果）

    调用 `MockTravelAgent`，快速返回预设的示例行程，主要用于调试前端调用链或演示流程，
    不依赖外部 API，也不会写入持久化任务状态。
    """
    try:
        # 生成测试任务ID
        task_id = str(uuid.uuid4())
        api_logger.info(f"模拟任务 {task_id}: 开始")

        # 计算旅行天数
        from datetime import datetime
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        duration = (end_date - start_date).days + 1
        if duration <= 0:
            raise HTTPException(status_code=400, detail="结束日期必须晚于或等于开始日期")

        # 转换请求为字典
        travel_request = request.model_dump()
        travel_request["duration"] = duration

        # 使用模拟智能体
        mock_agent = MockTravelAgent()
        result = mock_agent.run_travel_planning(travel_request)

        api_logger.info(f"模拟任务 {task_id}: 完成")

        return {
            "task_id": task_id,
            "status": "completed",
            "message": "模拟规划完成",
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模拟规划失败: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    自然语言交互接口 - 旅小智智能对话
    
    支持用户使用自然语言描述旅行需求，AI 自动提取关键信息并创建规划任务。
    
    示例输入：
    - "我想下周去北京玩3天，预算3000元，喜欢历史文化"
    - "帮我规划一个杭州5日游，2个人，预算中等"
    - "8月份去成都，想吃美食和看大熊猫"
    """
    try:
        user_message = request.message
        api_logger.info(f"收到自然语言请求: {user_message}")
        
        # 使用 LLM 解析用户意图
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            temperature=0.3
        )
        
        # 构造提示词
        system_prompt = """你是"旅小智"，一个专业的AI旅行规划助手。
你的任务是从用户的自然语言描述中提取旅行规划的关键信息。

请从用户输入中提取以下信息（如果有的话）：
1. destination: 目的地城市
2. start_date: 出发日期（格式：YYYY-MM-DD）
3. end_date: 返回日期（格式：YYYY-MM-DD）
4. duration: 旅行天数
5. budget_range: 预算范围（经济型/中等预算/豪华型）
6. group_size: 人数
7. interests: 兴趣爱好列表（如：美食、历史、自然风光等）

请返回 JSON 格式，包含：
- extracted: 提取到的信息字典
- missing: 缺失的关键信息列表
- confidence: 理解的置信度（0-1）
- clarification: 需要用户澄清的问题（如果有）

关键信息包括：destination（目的地）、时间信息（start_date/end_date/duration 至少一个）

如果用户没有提供具体日期，但提到了"下周"、"月底"、"国庆"等时间描述，请在 clarification 中询问具体日期。"""
        
        # 调用 LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户说：{user_message}\n\n今天是 {datetime.now().strftime('%Y年%m月%d日')}")
        ]
        
        response = llm.invoke(messages)
        
        # 解析 LLM 响应
        import json
        import re
        
        # 尝试从响应中提取 JSON
        response_text = response.content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            parsed_data = json.loads(json_match.group())
        else:
            # 如果没有找到JSON，返回错误
            return ChatResponse(
                understood=False,
                extracted_info={},
                missing_info=["所有信息"],
                clarification="抱歉，我没有理解您的需求。能否请您详细描述一下您的旅行计划？比如：目的地、时间、预算等。",
                can_proceed=False
            )
        
        extracted = parsed_data.get("extracted", {})
        missing = parsed_data.get("missing", [])
        confidence = parsed_data.get("confidence", 0.5)
        clarification_text = parsed_data.get("clarification", "")
        
        # 判断是否可以创建任务
        has_destination = "destination" in extracted and extracted["destination"]
        has_time_info = any(k in extracted for k in ["start_date", "end_date", "duration"])
        
        can_proceed = has_destination and has_time_info and confidence > 0.6
        
        # 如果可以创建任务，自动创建
        task_id = None
        if can_proceed:
            try:
                # 补充默认值
                travel_data = {
                    "destination": extracted.get("destination", ""),
                    "start_date": extracted.get("start_date", (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")),
                    "end_date": extracted.get("end_date", ""),
                    "budget_range": extracted.get("budget_range", "中等预算"),
                    "group_size": int(extracted.get("group_size", 2)),
                    "interests": extracted.get("interests", []),
                    "dietary_restrictions": "",
                    "activity_level": "适中",
                    "travel_style": "探索者",
                    "transportation_preference": "混合交通",
                    "accommodation_preference": "酒店",
                    "special_requirements": "",
                    "currency": "CNY"
                }
                
                # 处理日期
                if not travel_data["end_date"] and "duration" in extracted:
                    start_date_obj = datetime.strptime(travel_data["start_date"], "%Y-%m-%d")
                    duration_days = int(extracted["duration"])
                    end_date_obj = start_date_obj + timedelta(days=duration_days - 1)
                    travel_data["end_date"] = end_date_obj.strftime("%Y-%m-%d")
                elif not travel_data["end_date"]:
                    # 默认3天
                    start_date_obj = datetime.strptime(travel_data["start_date"], "%Y-%m-%d")
                    travel_data["end_date"] = (start_date_obj + timedelta(days=2)).strftime("%Y-%m-%d")
                
                # 计算天数
                start_date_obj = datetime.strptime(travel_data["start_date"], "%Y-%m-%d")
                end_date_obj = datetime.strptime(travel_data["end_date"], "%Y-%m-%d")
                duration = (end_date_obj - start_date_obj).days + 1
                if duration <= 0:
                    raise ValueError("聊天解析出的日期无效：结束日期早于开始日期")
                travel_data["duration"] = duration
                
                # 创建任务
                task_id = str(uuid.uuid4())
                planning_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "started",
                    "progress": 0,
                    "current_agent": "旅小智",
                    "message": f"旅小智正在为您规划{travel_data['destination']}之旅...",
                    "created_at": datetime.now().isoformat(),
                    "request": travel_data,
                    "result": None,
                    "source": "chat"  # 标记来源
                }
                
                # 保存任务状态
                save_tasks_state()
                
                # 添加后台任务
                background_tasks.add_task(run_planning_task, task_id, travel_data)
                
                api_logger.info(f"自然语言创建任务成功: {task_id}")
                
            except Exception as e:
                api_logger.error(f"自动创建任务失败: {str(e)}")
                can_proceed = False
        
        # 生成友好的反馈
        if can_proceed and task_id:
            clarification_response = f"✅ 好的！旅小智已经理解您的需求，正在为您规划{extracted.get('destination', '')}之旅！\n\n📋 规划信息：\n"
            if "destination" in extracted:
                clarification_response += f"📍 目的地：{extracted['destination']}\n"
            if "start_date" in extracted or "end_date" in extracted:
                clarification_response += f"📅 时间：{extracted.get('start_date', '')} 至 {extracted.get('end_date', '')}\n"
            if "duration" in extracted:
                clarification_response += f"⏰ 天数：{extracted['duration']}天\n"
            if "group_size" in extracted:
                clarification_response += f"👥 人数：{extracted['group_size']}人\n"
            if "budget_range" in extracted:
                clarification_response += f"💰 预算：{extracted['budget_range']}\n"
            if "interests" in extracted and extracted["interests"]:
                clarification_response += f"🎯 兴趣：{', '.join(extracted['interests'])}\n"
            
            clarification_response += "\n🤖 AI智能体团队正在为您工作，请稍候..."
        else:
            if not has_destination:
                clarification_response = "😊 您好！我是旅小智。请告诉我您想去哪里旅行？"
            elif not has_time_info:
                clarification_response = f"好的！您想去{extracted.get('destination', '')}旅行。\n\n请问您计划什么时候出发？大概玩几天呢？"
            else:
                clarification_response = clarification_text or "我需要更多信息来为您规划完美的旅程。"
            
            if missing:
                clarification_response += f"\n\n💡 还需要了解：{', '.join(missing)}"
        
        return ChatResponse(
            understood=confidence > 0.5,
            extracted_info=extracted,
            missing_info=missing,
            clarification=clarification_response,
            can_proceed=can_proceed,
            task_id=task_id
        )
        
    except Exception as e:
        api_logger.error(f"自然语言处理失败: {str(e)}")
        return ChatResponse(
            understood=False,
            extracted_info={},
            missing_info=["所有信息"],
            clarification="抱歉，旅小智遇到了一点小问题。能否请您重新描述一下您的旅行需求？",
            can_proceed=False
        )

# --------------------------- 独立运行入口 ---------------------------
if __name__ == "__main__":
    api_logger.info("启动AI旅行规划智能体API服务器…")
    api_logger.info("API文档: http://localhost:8080/docs")
    api_logger.info("健康检查: http://localhost:8080/health")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",  # 监听所有接口
        port=8080,
        reload=False,  # 禁用热重载，避免任务数据丢失
        log_level="info",
        timeout_keep_alive=30,  # 增加keep-alive超时
        timeout_graceful_shutdown=30,  # 优雅关闭超时
        access_log=True  # 启用访问日志
    )
