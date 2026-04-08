"""
LangGraph旅行规划智能体系统

这个模块实现了基于LangGraph框架的多智能体旅行规划系统。
它使用 OpenAI 兼容的大语言模型，通过多个专业智能体
的协作来生成全面的旅行计划。

主要组件：
1. TravelPlanState - 定义智能体间共享的状态结构
2. LangGraphTravelAgents - 主要的多智能体系统类
3. 各种专业智能体方法 - 每个智能体负责特定的规划任务

适用于大模型技术初级用户：
- LangGraph是一个用于构建多智能体系统的框架
- StateGraph管理智能体间的状态流转
- 每个智能体都是一个专门的函数，处理特定的任务
- 智能体通过共享状态进行通信和协作
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Callable
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
# 添加backend目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.langgraph_config import langgraph_config as config

# --------------------------- 日志配置 ---------------------------
def setup_agents_logger():
    logger = logging.getLogger('langgraph_agents')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = Path(__file__).resolve().parents[1] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "backend.log", encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

agents_logger = setup_agents_logger()

# 定义多智能体系统的状态结构
class TravelPlanState(TypedDict):
    """
    旅行规划状态类

    这个类定义了所有智能体共享的状态结构，包含了
    旅行规划过程中需要的所有信息。

    属性说明：
    - messages: 智能体间的消息历史
    - destination: 目的地
    - duration: 旅行天数
    - budget_range: 预算范围
    - interests: 兴趣爱好列表
    - group_size: 团队人数
    - travel_dates: 旅行日期
    - current_agent: 当前活跃的智能体
    - agent_outputs: 各智能体的输出结果
    - final_plan: 最终的旅行计划
    - iteration_count: 迭代次数
    """
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], add_messages]
    destination: str
    duration: int
    budget_range: str
    interests: List[str]
    group_size: int
    travel_dates: str
    current_agent: str
    agent_outputs: Dict[str, Any]
    final_plan: Dict[str, Any]
    iteration_count: int

class LangGraphTravelAgents:
    """
    基于LangGraph的多智能体旅行规划系统

    这个类是整个多智能体系统的核心，它：
    1. 初始化 OpenAI 兼容大语言模型
    2. 创建和管理智能体工作流图
    3. 协调各个专业智能体的工作
    4. 处理智能体间的状态传递和消息通信

    适用于大模型技术初级用户：
    这个类展示了如何使用LangGraph框架构建复杂的
    多智能体系统，每个智能体都有专门的职责。
    """

    def __init__(self):
        """
        初始化LangGraph旅行智能体系统

        配置 OpenAI 兼容大语言模型并创建智能体工作流图
        """
        # 初始化 OpenAI 兼容大语言模型
        self.llm_config = config.get_llm_config()
        self.llm = ChatOpenAI(**self.llm_config)

        # 初始化智能体工作流图
        self.graph = self._create_agent_graph()

    def _new_llm(self) -> ChatOpenAI:
        """创建一个新的 LLM 实例，避免并发线程复用同一实例造成的状态污染。"""
        return ChatOpenAI(**self.llm_config)

    @staticmethod
    def _safe_event_emit(
        event_callback: Optional[Callable[[Dict[str, Any]], None]],
        *,
        event_type: str,
        message: str,
        progress: Optional[int] = None,
        agent: Optional[str] = None,
        status: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not event_callback:
            return
        payload: Dict[str, Any] = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if progress is not None:
            payload["progress"] = progress
        if agent:
            payload["agent"] = agent
        if status:
            payload["status"] = status
        if data is not None:
            payload["data"] = data
        try:
            event_callback(payload)
        except Exception as callback_err:  # pragma: no cover - callback should never block planning
            agents_logger.warning(f"[EventCallback] 事件回调失败: {callback_err}")

    @staticmethod
    def _analysis_agents() -> List[str]:
        return ["travel_advisor", "weather_analyst", "budget_optimizer", "local_expert"]

    @staticmethod
    def _build_short_term_memory(travel_request: Dict[str, Any]) -> Dict[str, Any]:
        """构建会话级短期记忆（仅当前任务生命周期内有效）。"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        interests = travel_request.get("interests", [])
        if not isinstance(interests, list):
            interests = [str(interests)] if interests else []

        shared_facts = {
            "destination": str(travel_request.get("destination", "")).strip(),
            "duration": int(travel_request.get("duration", 3) or 3),
            "budget_range": str(travel_request.get("budget_range", "中等预算")),
            "interests": interests,
            "group_size": int(travel_request.get("group_size", 1) or 1),
            "travel_dates": str(travel_request.get("travel_dates", "")),
            "transportation_preference": str(travel_request.get("transportation_preference", "公共交通")),
            "accommodation_preference": str(travel_request.get("accommodation_preference", "酒店")),
        }

        agent_slots = {
            agent: {
                "status": "pending",
                "subtask": "",
                "agent_messages": [],
                "tool_artifacts": [],
                "output": "",
                "error": "",
                "started_at": "",
                "finished_at": "",
            }
            for agent in LangGraphTravelAgents._required_agents()
        }

        return {
            "session_id": session_id,
            "request_snapshot": travel_request,
            "shared_facts": shared_facts,
            "agent_slots": agent_slots,
            "tool_artifacts": [],
            "merge_notes": [],
            "timeline": [],
            "coordinator_plan": {},
            "collector_output": {},
            "itinerary_output": "",
            "coordinator_final_output": "",
            "final_output": "",
        }

    @staticmethod
    def _coordinator_skill_plan_subtasks(memory: Dict[str, Any]) -> Dict[str, str]:
        """协调员 skill：一次性拆分子任务，供并发阶段执行。"""
        facts = memory.get("shared_facts", {})
        destination = facts.get("destination", "目的地")
        duration = facts.get("duration", 3)
        budget = facts.get("budget_range", "中等预算")
        group_size = facts.get("group_size", 1)
        interests = "、".join(facts.get("interests", [])) or "无"
        travel_dates = facts.get("travel_dates", "未指定")

        return {
            "travel_advisor": (
                f"围绕{destination}给出{duration}天的玩法策略，聚焦与兴趣[{interests}]相关的必去与可选景点，"
                "同时给出区域组织建议与适合人群说明。"
            ),
            "weather_analyst": (
                f"分析{destination}在{travel_dates}期间天气，给出活动时段建议、天气风险提醒与行李建议，"
                "并指出对户外/室内安排的影响。"
            ),
            "budget_optimizer": (
                f"基于{destination}、{duration}天、{group_size}人、预算[{budget}]输出预算拆分与省钱策略，"
                "覆盖住宿、交通、餐饮、活动四类。"
            ),
            "local_expert": (
                f"产出{destination}在地建议，重点给出小众地点、文化礼仪、本地餐饮、避坑建议。"
                "优先引用本地专家 skill/RAG 证据。"
            ),
            "itinerary_planner": (
                "在汇总前四个分析智能体结果后，输出逐日行程，要求时间段清晰、路线顺畅、兼顾预算与天气约束。"
            ),
        }

    def _run_analysis_agent_with_private_context(
        self,
        agent_name: str,
        subtask: str,
        memory: Dict[str, Any],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """分析型 Agent 的私有上下文执行闭环：工具调用 + 总结输出。"""
        facts = memory.get("shared_facts", {})
        destination = str(facts.get("destination", "")).strip()
        interests_text = "、".join(facts.get("interests", []))
        duration = int(facts.get("duration", 3))
        budget_range = str(facts.get("budget_range", "中等预算"))
        group_size = int(facts.get("group_size", 1))
        travel_dates = str(facts.get("travel_dates", ""))

        llm = self._new_llm()
        started_at = datetime.now().isoformat()
        tool_artifacts: List[Dict[str, Any]] = []
        agent_messages: List[str] = []

        def add_tool_artifact(
            tool_name: str,
            params: Dict[str, Any],
            result_text: Any,
            *,
            success: bool = True,
        ) -> None:
            result = str(result_text)
            artifact = {
                "agent": agent_name,
                "tool": tool_name,
                "params": params,
                "result": result,
                "result_preview": result[:300],
                "status": "completed" if success else "failed",
                "timestamp": datetime.now().isoformat(),
            }
            tool_artifacts.append(artifact)
            self._safe_event_emit(
                event_callback,
                event_type="tool_completed" if success else "tool_failed",
                message=f"{agent_name} 工具调用完成: {tool_name}" if success else f"{agent_name} 工具调用失败: {tool_name}",
                agent=agent_name,
                data={
                    "tool": tool_name,
                    "preview": artifact["result_preview"],
                    "status": artifact["status"],
                },
            )

        def run_tool(tool_name: str, tool_callable: Any, params: Dict[str, Any], *, async_mode: bool = False) -> Any:
            self._safe_event_emit(
                event_callback,
                event_type="tool_called",
                message=f"{agent_name} 调用工具: {tool_name}",
                agent=agent_name,
                data={"tool": tool_name, "params": params},
            )
            try:
                if async_mode:
                    result = asyncio.run(tool_callable.ainvoke(params))
                else:
                    result = tool_callable.invoke(params)
                add_tool_artifact(tool_name, params, result, success=True)
                return result
            except Exception as tool_err:
                fallback_text = f"{tool_name} 调用失败: {tool_err}"
                add_tool_artifact(tool_name, params, fallback_text, success=False)
                agent_messages.append(f"tool_error[{tool_name}]={tool_err}")
                return fallback_text

        try:
            from tools.travel_tools import (
                search_destination_info,
                search_weather_info,
                search_attractions,
                local_expert_skill,
                search_budget_info,
            )

            if agent_name == "travel_advisor":
                params_1 = {"query": f"{destination} 旅游目的地 玩法 推荐"}
                run_tool("search_destination_info", search_destination_info, params_1)

                params_2 = {"destination": destination, "interests": interests_text}
                run_tool("search_attractions", search_attractions, params_2)

            elif agent_name == "weather_analyst":
                params = {"destination": destination, "dates": travel_dates}
                run_tool("search_weather_info", search_weather_info, params, async_mode=True)

            elif agent_name == "budget_optimizer":
                params = {
                    "destination": destination,
                    "duration": str(duration),
                    "budget_range": budget_range,
                    "group_size": str(group_size),
                }
                run_tool("search_budget_info", search_budget_info, params)

            elif agent_name == "local_expert":
                params = {
                    "destination": destination,
                    "interests": interests_text,
                    "query": f"{destination} 在地建议 小众地点 文化礼仪 本地餐饮 避坑",
                    "top_k": 4,
                }
                run_tool("local_expert_skill", local_expert_skill, params)

            tool_context = "\n\n".join(
                [
                    f"[{idx + 1}] {item['tool']}:\n{item['result']}"
                    for idx, item in enumerate(tool_artifacts)
                ]
            )
            agent_messages.append(f"subtask={subtask}")
            agent_messages.append(f"tool_count={len(tool_artifacts)}")

            system_prompt = (
                f"你是{agent_name}智能体。你正在共享会话下的并发执行阶段工作。\n"
                "请严格基于提供的工具证据输出结论，不要编造。\n"
                "输出要求：\n"
                "1) 先给出3-6条可执行建议\n"
                "2) 再给出关键注意事项\n"
                "3) 若存在不确定信息，请明确标注“待确认”\n"
                "4) 使用中文，结构清晰\n"
            )
            user_prompt = (
                f"【子任务】\n{subtask}\n\n"
                f"【会话事实】\n"
                f"- 目的地: {destination}\n"
                f"- 行程天数: {duration}\n"
                f"- 预算: {budget_range}\n"
                f"- 人数: {group_size}\n"
                f"- 兴趣: {interests_text or '无'}\n"
                f"- 日期: {travel_dates or '未指定'}\n\n"
                f"【工具证据】\n{tool_context or '无工具证据'}\n"
            )
            error_text = ""
            try:
                response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                output_text = str(response.content).strip()
                if not output_text:
                    raise RuntimeError("LLM 返回空内容")
            except Exception as llm_err:
                output_text = self._build_analysis_fallback_output(
                    agent_name=agent_name,
                    subtask=subtask,
                    facts=facts,
                    tool_artifacts=tool_artifacts,
                    error_reason=str(llm_err),
                )
                error_text = f"LLM调用失败，已降级输出: {llm_err}"
                agent_messages.append(f"llm_fallback={llm_err}")

            return {
                "status": "completed",
                "subtask": subtask,
                "agent_messages": agent_messages,
                "tool_artifacts": tool_artifacts,
                "output": output_text,
                "error": error_text,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
            }
        except Exception as agent_err:
            fallback_output = self._build_analysis_fallback_output(
                agent_name=agent_name,
                subtask=subtask,
                facts=facts,
                tool_artifacts=tool_artifacts,
                error_reason=str(agent_err),
            )
            return {
                "status": "completed",
                "subtask": subtask,
                "agent_messages": agent_messages,
                "tool_artifacts": tool_artifacts,
                "output": fallback_output,
                "error": f"Agent执行异常，已降级输出: {agent_err}",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
            }

    @staticmethod
    def _build_analysis_fallback_output(
        agent_name: str,
        subtask: str,
        facts: Dict[str, Any],
        tool_artifacts: List[Dict[str, Any]],
        error_reason: str,
    ) -> str:
        """分析型 Agent 在外部服务波动时的稳定降级输出。"""
        destination = str(facts.get("destination", "目的地"))
        duration = int(facts.get("duration", 3) or 3)
        budget = str(facts.get("budget_range", "中等预算"))
        interests = "、".join(facts.get("interests", [])) or "无"
        travel_dates = str(facts.get("travel_dates", "未指定"))

        evidence: List[str] = []
        for item in tool_artifacts[:3]:
            tool_name = str(item.get("tool", "tool"))
            status = str(item.get("status", "unknown"))
            preview = str(item.get("result_preview", "")).replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:100] + "..."
            evidence.append(f"- {tool_name}({status}): {preview or '无返回'}")
        if not evidence:
            evidence.append("- 当前未获取到可靠外部检索结果，建议出发前二次确认。")

        suggestions: List[str]
        caution = "请在出行前 24 小时复核天气、交通与门票信息。"
        if agent_name == "travel_advisor":
            suggestions = [
                f"以{destination}核心区域为主线，优先安排 2-3 个高价值景点，避免单日跨区折返。",
                f"按{duration}天拆分节奏：上午核心景点、下午体验项目、晚上本地街区漫游。",
                f"结合兴趣[{interests}]设置必做清单与备选清单，天气变化时优先切换室内项目。",
            ]
        elif agent_name == "weather_analyst":
            suggestions = [
                f"将{travel_dates}期间的户外活动安排在体感较舒适时段，保留室内备选计划。",
                "行李建议按“轻便分层+防晒/防雨”准备，减少天气突变影响。",
                "若出现降雨或强风，优先调整为博物馆、商圈、展馆等室内线路。",
            ]
            caution = "天气服务当前存在波动，最终安排请以官方临近预报为准。"
        elif agent_name == "budget_optimizer":
            suggestions = [
                f"按预算[{budget}]拆分为住宿/交通/餐饮/活动四类，先锁定住宿上限再分配弹性支出。",
                "热门景点门票与交通尽量提前预订，优先可退改方案以降低不确定成本。",
                f"{duration}天行程建议预留约 10%-15% 机动预算，覆盖临时改签与排队替代成本。",
            ]
        else:
            suggestions = [
                f"围绕{destination}优先选择口碑稳定、通勤友好的在地街区进行深度体验。",
                "餐饮选择遵循“错峰+就近+高评分近三月评价”原则，减少等待与踩坑概率。",
                "文化礼仪与排队秩序优先遵守本地惯例，避免高峰期临时打车带来的时间损失。",
            ]

        lines = [
            f"## {agent_name} 分析（降级模式）",
            f"- 子任务: {subtask}",
            f"- 目的地: {destination}",
            f"- 行程天数: {duration}",
            f"- 预算范围: {budget}",
            f"- 兴趣偏好: {interests}",
            f"- 触发原因: {error_reason}",
            "",
            "建议：",
        ]
        lines.extend([f"{idx + 1}. {item}" for idx, item in enumerate(suggestions)])
        lines.extend(["", "证据摘要："])
        lines.extend(evidence)
        lines.extend(["", f"注意事项：{caution}（标记：待确认）"])
        return "\n".join(lines)

    def _collector_stage(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """汇总并规范化并发分析结果。"""
        slots = memory.get("agent_slots", {})
        sections: Dict[str, str] = {}
        missing: List[str] = []
        failures: List[str] = []
        notes: List[str] = []

        for agent in self._analysis_agents():
            slot = slots.get(agent, {})
            status = str(slot.get("status", "pending"))
            output = str(slot.get("output", "")).strip()

            if status == "completed" and output:
                sections[agent] = output
            elif status == "failed":
                failures.append(agent)
                missing.append(agent)
                notes.append(f"{agent} 执行失败: {slot.get('error', '未知错误')}")
            else:
                missing.append(agent)
                notes.append(f"{agent} 未生成可用输出。")

        collector = {
            "sections": sections,
            "missing_agents": missing,
            "failed_agents": failures,
            "merge_notes": notes,
            "generated_at": datetime.now().isoformat(),
        }
        memory["merge_notes"].extend(notes)
        memory["collector_output"] = collector
        return collector

    def _itinerary_planner_stage(self, memory: Dict[str, Any], collector_output: Dict[str, Any]) -> str:
        """串行执行 itinerary_planner，基于汇总结果生成每日行程。"""
        facts = memory.get("shared_facts", {})
        destination = facts.get("destination", "")
        duration = facts.get("duration", 3)
        budget_range = facts.get("budget_range", "中等预算")
        travel_dates = facts.get("travel_dates", "")
        interests = "、".join(facts.get("interests", [])) or "无"
        group_size = facts.get("group_size", 1)

        sections = collector_output.get("sections", {})
        merged_context = "\n\n".join([f"【{name}】\n{content}" for name, content in sections.items()])

        llm = self._new_llm()
        system_prompt = (
            "你是 itinerary_planner 智能体。请基于多智能体汇总结果生成可执行的逐日旅行计划。"
            "计划必须考虑天气、预算、本地建议与兴趣偏好。"
        )
        user_prompt = (
            f"- 目的地: {destination}\n"
            f"- 天数: {duration}\n"
            f"- 日期: {travel_dates}\n"
            f"- 人数: {group_size}\n"
            f"- 预算: {budget_range}\n"
            f"- 兴趣: {interests}\n\n"
            "请按“Day1/Day2...”输出：\n"
            "1. 上午/下午/晚上安排\n"
            "2. 交通与距离建议\n"
            "3. 餐饮建议\n"
            "4. 费用与风险提示\n\n"
            f"多智能体汇总输入：\n{merged_context or '无'}"
        )
        try:
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            text = str(response.content).strip()
            if not text:
                raise RuntimeError("itinerary_planner 返回空内容")
        except Exception as planner_err:
            memory.setdefault("merge_notes", []).append(
                f"itinerary_planner 调用失败，已降级模板生成: {planner_err}"
            )
            text = self._build_fallback_itinerary(facts, collector_output, str(planner_err))
        memory["itinerary_output"] = text
        return text

    def _coordinator_finalize_stage(
        self,
        memory: Dict[str, Any],
        collector_output: Dict[str, Any],
        itinerary_output: str,
    ) -> str:
        """
        协调员终审阶段：
        - 接收 itinerary_planner 的整合结果
        - 结合并发分析输出进行一致性检查
        - 产出可直接进入最终落库的协调员结论
        """
        facts = memory.get("shared_facts", {})
        destination = facts.get("destination", "")
        duration = facts.get("duration", 3)
        budget_range = facts.get("budget_range", "中等预算")
        travel_dates = facts.get("travel_dates", "")
        interests = "、".join(facts.get("interests", [])) or "无"

        sections = collector_output.get("sections", {})
        missing_agents = collector_output.get("missing_agents", [])
        merged_context = "\n\n".join([f"【{name}】\n{content}" for name, content in sections.items()])

        llm = self._new_llm()
        system_prompt = (
            "你是 coordinator 智能体，负责最终审阅。"
            "你必须基于已完成的 itinerary 以及并发分析证据，输出最终协调结论。"
            "输出要包含：一致性检查、关键取舍、风险提示、可执行结论。"
        )
        user_prompt = (
            f"- 目的地: {destination}\n"
            f"- 天数: {duration}\n"
            f"- 日期: {travel_dates}\n"
            f"- 预算: {budget_range}\n"
            f"- 兴趣: {interests}\n"
            f"- 缺失智能体: {', '.join(missing_agents) if missing_agents else '无'}\n\n"
            f"【并发分析汇总】\n{merged_context or '无'}\n\n"
            f"【itinerary_planner 输出】\n{itinerary_output or '无'}\n"
        )

        try:
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            text = str(response.content).strip()
            if not text:
                raise RuntimeError("coordinator 终审返回空内容")
        except Exception as coordinator_err:
            text = self._build_coordinator_fallback_output(
                facts=facts,
                collector_output=collector_output,
                itinerary_output=itinerary_output,
                error_reason=str(coordinator_err),
            )
            memory.setdefault("merge_notes", []).append(
                f"coordinator 终审调用失败，已降级模板生成: {coordinator_err}"
            )

        memory["coordinator_final_output"] = text
        return text

    @staticmethod
    def _build_fallback_itinerary(
        facts: Dict[str, Any],
        collector_output: Dict[str, Any],
        error_reason: str,
    ) -> str:
        """在 LLM 不可用时，生成可读的基础行程兜底文本。"""
        destination = str(facts.get("destination", "目的地"))
        duration = int(facts.get("duration", 3) or 3)
        duration = max(1, min(duration, 15))
        travel_dates = str(facts.get("travel_dates", "未指定"))
        budget_range = str(facts.get("budget_range", "中等预算"))
        interests = "、".join(facts.get("interests", [])) or "无"
        sections = collector_output.get("sections", {}) if isinstance(collector_output, dict) else {}

        lines: List[str] = []
        lines.append(f"## {destination} 行程（降级兜底版）")
        lines.append(f"- 日期范围: {travel_dates}")
        lines.append(f"- 预算范围: {budget_range}")
        lines.append(f"- 兴趣偏好: {interests}")
        lines.append(f"- 生成方式: 模板兜底（原因: {error_reason}）")
        lines.append("")

        advisor = str(sections.get("travel_advisor", "请优先参考经典景点与城市地标。")).strip()
        weather = str(sections.get("weather_analyst", "天气信息获取受限，请出行前再次确认预报。")).strip()
        budget = str(sections.get("budget_optimizer", "建议按住宿/交通/餐饮/门票拆分预算。")).strip()
        local = str(sections.get("local_expert", "请遵循在地礼仪，优先热门安全区域活动。")).strip()

        advisor_hint = advisor[:120] + ("..." if len(advisor) > 120 else "")
        weather_hint = weather[:120] + ("..." if len(weather) > 120 else "")
        budget_hint = budget[:120] + ("..." if len(budget) > 120 else "")
        local_hint = local[:120] + ("..." if len(local) > 120 else "")

        for day in range(1, duration + 1):
            lines.append(f"### Day{day}")
            lines.append("- 上午: 城市核心区域步行/地标打卡，尽量减少跨区通勤。")
            lines.append("- 下午: 结合博物馆/室内点位与特色街区，灵活避开拥堵时段。")
            lines.append("- 晚上: 本地餐饮体验 + 休闲夜景，控制当日总预算。")
            lines.append(f"- 参考线索(顾问): {advisor_hint}")
            lines.append(f"- 参考线索(天气): {weather_hint}")
            lines.append(f"- 参考线索(预算): {budget_hint}")
            lines.append(f"- 参考线索(在地): {local_hint}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _build_coordinator_fallback_output(
        facts: Dict[str, Any],
        collector_output: Dict[str, Any],
        itinerary_output: str,
        error_reason: str,
    ) -> str:
        """协调员终审降级输出，确保数据流完整。"""
        destination = str(facts.get("destination", "目的地"))
        budget_range = str(facts.get("budget_range", "中等预算"))
        interests = "、".join(facts.get("interests", [])) or "无"
        missing_agents = (
            collector_output.get("missing_agents", [])
            if isinstance(collector_output, dict)
            else []
        )
        merge_notes = (
            collector_output.get("merge_notes", [])
            if isinstance(collector_output, dict)
            else []
        )
        notes_text = "\n".join([f"- {note}" for note in merge_notes]) if merge_notes else "- 无"
        missing_text = "、".join(missing_agents) if missing_agents else "无"

        return (
            f"## 协调员终审结论（降级兜底）\n"
            f"- 目的地: {destination}\n"
            f"- 预算约束: {budget_range}\n"
            f"- 兴趣偏好: {interests}\n"
            f"- 缺失智能体: {missing_text}\n"
            f"- 生成方式: 模板兜底（原因: {error_reason}）\n\n"
            "### 一致性检查\n"
            "- 已基于并发分析输出与 itinerary_planner 结果进行最终合并。\n"
            "- 若存在缺失智能体，请优先补齐对应维度后再二次确认。\n\n"
            "### 合并备注\n"
            f"{notes_text}\n\n"
            "### 可执行行程（来自 itinerary_planner）\n"
            f"{itinerary_output or '暂无可用行程内容。'}"
        ).strip()

    @staticmethod
    def _final_summarizer_stage(
        memory: Dict[str, Any],
        collector_output: Dict[str, Any],
        itinerary_output: str,
        coordinator_output: str,
    ) -> str:
        """最终整合输出。"""
        facts = memory.get("shared_facts", {})
        destination = facts.get("destination", "")
        duration = facts.get("duration", 3)
        budget_range = facts.get("budget_range", "中等预算")
        travel_dates = facts.get("travel_dates", "")
        group_size = facts.get("group_size", 1)
        interests = "、".join(facts.get("interests", [])) or "无"

        notes = collector_output.get("merge_notes", [])
        note_text = "\n".join([f"- {n}" for n in notes]) if notes else "- 无"

        summary = (
            f"# {destination}旅行方案（并发多Agent版）\n\n"
            f"- 行程天数: {duration} 天\n"
            f"- 出行人数: {group_size} 人\n"
            f"- 预算范围: {budget_range}\n"
            f"- 日期: {travel_dates}\n"
            f"- 兴趣: {interests}\n\n"
            "## 执行说明\n"
            "- 本方案采用“共享会话 + Agent私有上下文 + 并发分析 + 串行整合”架构生成。\n"
            "- 本地专家保留 skill + RAG 检索策略。\n\n"
            "## 关键合并备注\n"
            f"{note_text}\n\n"
            "## 协调员终审结论\n"
            f"{coordinator_output or '暂无协调员终审内容。'}\n\n"
            "## 逐日行程建议\n"
            f"{itinerary_output or '暂无可用行程内容。'}\n"
        )
        memory["final_output"] = summary
        return summary

    @staticmethod
    def _build_agent_slot_status_snapshot(memory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """提取短期记忆所需的 Agent 状态快照。"""
        snapshot: Dict[str, Dict[str, Any]] = {}
        slots = memory.get("agent_slots", {}) if isinstance(memory, dict) else {}
        for agent_name, slot in slots.items():
            slot_dict = slot if isinstance(slot, dict) else {}
            output_text = str(slot_dict.get("output", "")).strip()
            snapshot[agent_name] = {
                "status": str(slot_dict.get("status", "pending")),
                "subtask": str(slot_dict.get("subtask", "")),
                "error": str(slot_dict.get("error", "")),
                "started_at": str(slot_dict.get("started_at", "")),
                "finished_at": str(slot_dict.get("finished_at", "")),
                "tool_count": len(slot_dict.get("tool_artifacts", []) or []),
                "has_output": bool(output_text),
                "output_preview": output_text[:200],
            }
        return snapshot

    @staticmethod
    def _required_agents() -> List[str]:
        return ["travel_advisor", "weather_analyst", "budget_optimizer", "local_expert", "itinerary_planner"]

    @staticmethod
    def _derive_agent_status(response_content: Any) -> str:
        text = response_content if isinstance(response_content, str) else str(response_content)
        return "needs_search" if "NEED_SEARCH:" in text else "completed"

    def _get_missing_agents(self, state: TravelPlanState) -> List[str]:
        required_agents = self._required_agents()
        agent_outputs = state.get("agent_outputs", {}) or {}
        missing_agents: List[str] = []

        for agent in required_agents:
            output = agent_outputs.get(agent)
            if not output:
                missing_agents.append(agent)
                continue

            status = str(output.get("status", "")).lower()
            response_text = str(output.get("response", ""))
            if status != "completed" or "NEED_SEARCH:" in response_text:
                missing_agents.append(agent)

        return missing_agents

    @staticmethod
    def _coordinator_decision_from_text(raw_text: str) -> str:
        content = (raw_text or "").strip().lower()
        if "travel_advisor" in content or "旅行顾问" in content:
            return "travel_advisor"
        if "weather_analyst" in content or "天气分析师" in content:
            return "weather_analyst"
        if "budget_optimizer" in content or "预算优化师" in content:
            return "budget_optimizer"
        if "local_expert" in content or "当地专家" in content:
            return "local_expert"
        if "itinerary_planner" in content or "行程规划师" in content:
            return "itinerary_planner"
        if "search" in content or "need_search" in content or "搜索" in content:
            return "search"
        if "final_plan" in content or "最终计划" in content:
            return "final_plan"
        return "unknown"

    def _create_agent_graph(self) -> StateGraph:
        """
        创建LangGraph多智能体工作流图

        解释：
        该方法负责构建整个多智能体系统的工作流图（StateGraph）。
        在LangGraph框架中，StateGraph用于定义各个智能体节点（如旅行顾问、天气分析师等）以及它们之间的连接关系和执行顺序。
        通过添加节点和设置条件边缘，可以灵活地控制智能体之间的协作流程，实现复杂的多智能体任务分工与协作。

        这个方法构建了智能体间的工作流程图，定义了：
        1. 各个智能体节点
        2. 智能体间的连接关系
        3. 工作流的执行顺序

        返回：配置好的StateGraph工作流对象
        """

        # 定义工作流图
        workflow = StateGraph(TravelPlanState)

        # 添加智能体节点
        workflow.add_node("travel_advisor", self._travel_advisor_agent)    # 旅行顾问
        workflow.add_node("weather_analyst", self._weather_analyst_agent)  # 天气分析师
        workflow.add_node("budget_optimizer", self._budget_optimizer_agent) # 预算优化师
        workflow.add_node("local_expert", self._local_expert_agent)        # 当地专家
        workflow.add_node("itinerary_planner", self._itinerary_planner_agent) # 行程规划师
        workflow.add_node("coordinator", self._coordinator_agent)             # 协调员
        workflow.add_node("tools", self._tool_executor_node)                  # 工具执行器

        # 定义工作流边缘（智能体间的连接）
        workflow.set_entry_point("coordinator")  # 设置协调员为入口点

        # 协调员决定调用哪些智能体
        workflow.add_conditional_edges(
            "coordinator",                    # 从协调员开始
            self._coordinator_router,         # 使用协调员路由器决定下一步
            {
                "travel_advisor": "travel_advisor",      # 可以转到旅行顾问
                "weather_analyst": "weather_analyst",    # 可以转到天气分析师
                "budget_optimizer": "budget_optimizer",  # 可以转到预算优化师
                "local_expert": "local_expert",          # 可以转到当地专家
                "itinerary_planner": "itinerary_planner", # 可以转到行程规划师
                "tools": "tools",                        # 可以转到工具执行器
                "end": END                               # 可以结束流程
            }
        )

        # 每个智能体都可以使用工具或返回协调员
        for agent in ["travel_advisor", "weather_analyst", "budget_optimizer", "local_expert", "itinerary_planner"]:
            workflow.add_conditional_edges(
                agent,                        # 从各个智能体
                self._agent_router,           # 使用智能体路由器决定下一步
                {
                    "tools": "tools",         # 可以转到工具执行器
                    "coordinator": "coordinator", # 可以返回协调员
                    "end": END               # 可以结束流程
                }
            )

        # 工具执行器总是返回协调员
        workflow.add_edge("tools", "coordinator")

        # 编译并返回工作流
        return workflow.compile()

    def _coordinator_agent(self, state: TravelPlanState) -> TravelPlanState:
        """
        协调员智能体 - 编排多智能体工作流

        协调员是整个系统的"大脑"，负责：
        1. 分析当前状态和需求
        2. 决定下一步需要哪个智能体工作
        3. 综合各智能体的输出
        4. 判断是否需要更多信息或可以结束

        参数：
        - state: 当前的旅行规划状态

        返回：更新后的状态
        """

        required_agents = self._required_agents()
        missing_agents = self._get_missing_agents(state)
        completed_agents = [agent for agent in required_agents if agent not in missing_agents]

        system_prompt = f"""您是多智能体旅行规划系统的协调员智能体。

您的职责是：
1. 分析旅行规划请求
2. 确定需要哪些专业智能体参与
3. 协调智能体间的工作流程
4. 综合最终建议

当前请求：
- 目的地: {state.get('destination', '未指定')}
- 时长: {state.get('duration', '未指定')} 天
- 预算: {state.get('budget_range', '未指定')}
- 兴趣: {', '.join(state.get('interests', []))}
- 团队人数: {state.get('group_size', 1)}
- 旅行日期: {state.get('travel_dates', '未指定')}

可用智能体：
- travel_advisor: 目的地专业知识和景点推荐
- weather_analyst: 天气预报和活动规划
- budget_optimizer: 成本分析和省钱策略
- local_expert: 本地洞察和文化贴士
- itinerary_planner: 日程优化和物流安排

目前智能体输出: {json.dumps(state.get('agent_outputs', {}), indent=2)}
已完成智能体: {completed_agents}
待完成智能体: {missing_agents}

根据当前状态，决定下一步行动：
1. 如果需要更多信息，指定下一个应该工作的智能体
2. 如果从所有相关智能体获得了足够信息，综合最终计划
3. 回应智能体名称或'FINAL_PLAN'（如果准备结束）

您的响应应该是以下之一：
- 下一个要调用的智能体名称 (travel_advisor, weather_analyst, budget_optimizer, local_expert, itinerary_planner)
- 'FINAL_PLAN' 如果准备创建综合旅行计划
- 'SEARCH' 如果需要先搜索信息

严格规则（必须遵守）：
1) 只输出一个标记，不要解释，不要换行，不要附加文字。
2) 如果“待完成智能体”非空，禁止输出 FINAL_PLAN。
3) 当待完成智能体非空时，优先输出待完成列表中的一个智能体名称。
4) 只有待完成智能体为空时，才允许输出 FINAL_PLAN。
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-3:])  # Keep recent context
        
        raw_response = self.llm.invoke(messages)
        raw_text = raw_response.content if isinstance(raw_response.content, str) else str(raw_response.content)
        decision = self._coordinator_decision_from_text(raw_text)

        agent_outputs = state.get("agent_outputs", {}) or {}
        pending_search_agents: List[str] = []
        for agent in missing_agents:
            output = agent_outputs.get(agent)
            if not isinstance(output, dict):
                continue
            status = str(output.get("status", "")).lower()
            if status == "needs_search":
                pending_search_agents.append(agent)

        if missing_agents:
            # 若某智能体已明确发起过工具检索（needs_search），优先回流该智能体完成“基于检索结果的二次分析”
            if pending_search_agents and decision not in pending_search_agents:
                decision = pending_search_agents[0]
            if decision in ("final_plan", "search", "unknown") or decision not in required_agents:
                decision = missing_agents[0]
        else:
            if decision in ("search", "unknown") or decision in required_agents:
                decision = "final_plan"

        agents_logger.info(
            f"[CoordinatorAgent] 原始输出: {raw_text} | 归一化决策: {decision} | 待完成: {missing_agents}"
        )
        response = AIMessage(content=decision)
        
        # Update state
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "coordinator"
        new_state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        return new_state
    
    def _travel_advisor_agent(self, state: TravelPlanState) -> TravelPlanState:
        """
        旅行顾问智能体，具有目的地专业知识

        这个智能体专门负责提供目的地相关的专业建议，
        包括景点推荐、文化洞察等。
        """

        system_prompt = f"""您是旅行顾问智能体，专门从事目的地专业知识和推荐服务。

您的专业领域包括：
- 目的地知识和亮点
- 景点推荐
- 文化洞察和贴士
- 旅行者优秀做法

当前规划请求：
- 目的地: {state.get('destination')}
- 时长: {state.get('duration')} 天
- 兴趣: {', '.join(state.get('interests', []))}
- 团队人数: {state.get('group_size')}

您的任务：提供全面的目的地建议，包括：
1. 顶级景点和必游之地
2. 文化洞察和礼仪贴士
3. 最佳住宿和探索区域
4. 基于兴趣的活动推荐

如果您需要搜索关于目的地的当前信息，请回复 'NEED_SEARCH: [搜索查询]'
否则，请基于您的知识提供专家建议。
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        
        response = self.llm.invoke(messages)
        
        # Store agent output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["travel_advisor"] = {
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "status": self._derive_agent_status(response.content)
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "travel_advisor"
        new_state["agent_outputs"] = agent_outputs
        
        return new_state
    
    def _weather_analyst_agent(self, state: TravelPlanState) -> TravelPlanState:
        """
        天气分析师智能体，专门进行气候和天气规划

        这个智能体专门负责天气情报分析和基于气候的
        活动规划建议。
        """

        destination = str(state.get("destination", "")).strip()
        recent_messages = state.get("messages", [])
        has_recent_weather_result = False
        for msg in reversed(recent_messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue
            lowered = content.lower()
            if "搜索结果: [search_weather_info]" in content:
                has_recent_weather_result = True
                break
            if "搜索结果:" in content and (
                "天气" in content
                or "weather" in lowered
                or "气温" in content
                or "温度" in content
                or "日出" in content
                or "日落" in content
            ):
                if not destination or destination in content:
                    has_recent_weather_result = True
                    break

        workflow_rule = (
            "⚠️ 重要工作流程：\n"
            "1. 当前已有天气搜索结果，禁止再次发起 NEED_SEARCH\n"
            "2. 请基于现有搜索结果直接输出完整天气分析结论\n"
        ) if has_recent_weather_result else (
            "⚠️ 重要工作流程：\n"
            "1. 【强制要求】您必须首先调用天气搜索工具获取实时准确的天气数据\n"
            f"2. 请立即回复：'NEED_SEARCH: {state.get('destination')} {state.get('travel_dates')} 天气预报'\n"
            "3. 获取天气数据后，再基于实际天气信息提供专业分析\n"
        )

        system_prompt = f"""您是天气分析师智能体，专门从事天气情报和气候感知规划。

        您的专业领域包括：
        - 天气模式分析
        - 季节性旅行推荐
        - 基于天气条件的活动规划
        - 目的地气候考虑因素

        当前规划请求：
        - 目的地: {state.get('destination')}
        - 旅行日期: {state.get('travel_dates')}
        - 时长: {state.get('duration')} 天
        - 计划活动: {', '.join(state.get('interests', []))}

        {workflow_rule}

        您的最终任务（在获取天气数据后）：
        1. 分析旅行日期期间的实际天气条件
        2. 基于真实天气数据推荐户外活动的最佳时间段
        3. 根据天气情况提供适合的活动建议
        4. 提供基于实际气候的打包建议

        注意：必须先获取实时天气数据，不要仅凭经验或历史气候知识进行推测。
        """
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-6:])
        
        response = self.llm.invoke(messages)
        
        # Store agent output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["weather_analyst"] = {
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "status": self._derive_agent_status(response.content)
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "weather_analyst"
        new_state["agent_outputs"] = agent_outputs
        
        return new_state
    
    def _budget_optimizer_agent(self, state: TravelPlanState) -> TravelPlanState:
        """
        预算优化师智能体，专门进行成本分析和优化

        这个智能体专门负责旅行预算的分析和优化，
        提供省钱策略和成本效益建议。
        """

        recent_messages = state.get("messages", [])
        has_recent_budget_result = False
        for msg in reversed(recent_messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue
            lowered = content.lower()
            if "搜索结果: [search_budget_info]" in content:
                has_recent_budget_result = True
                break
            if "搜索结果:" in content and (
                "预算" in content
                or "费用" in content
                or "cost" in lowered
                or "price" in lowered
                or "人均" in content
            ):
                has_recent_budget_result = True
                break

        workflow_rule = (
            "执行约束：\n"
            "1. 当前已有预算检索结果，禁止再次发起 NEED_SEARCH\n"
            "2. 必须基于现有结果输出预算分析结论\n"
        ) if has_recent_budget_result else (
            "执行约束：\n"
            "1. 当前尚无预算检索结果，必须先调用工具\n"
            f"2. 请先回复：'NEED_SEARCH: {state.get('destination')} {state.get('duration')}天 {state.get('group_size')}人 预算 费用 价格'\n"
            "3. 获取检索结果后，再输出预算分析结论\n"
        )

        system_prompt = f"""您是预算优化师智能体，专门从事成本分析和省钱策略。

您的专业领域包括：
- 旅行成本分析和预算制定
- 省钱贴士和策略
- 预算分配建议
- 经济实惠的替代方案

当前规划请求：
- 目的地: {state.get('destination')}
- 时长: {state.get('duration')} 天
- 预算范围: {state.get('budget_range')}
- 团队人数: {state.get('group_size')}

您的任务：提供预算优化建议，包括：
1. 估算每日和总费用
2. 按类别分解预算（住宿、餐饮、活动、交通）
3. 省钱贴士和策略
4. 昂贵活动的经济实惠替代方案

{workflow_rule}

如果您需要当前价格信息，请回复 'NEED_SEARCH: [预算搜索查询]'
否则，请提供您的预算分析和建议。
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-4:])
        
        response = self.llm.invoke(messages)
        
        # Store agent output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["budget_optimizer"] = {
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "status": self._derive_agent_status(response.content)
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "budget_optimizer"
        new_state["agent_outputs"] = agent_outputs
        
        return new_state
    
    def _local_expert_agent(self, state: TravelPlanState) -> TravelPlanState:
        """
        当地专家智能体，具有内部知识和本地洞察

        这个智能体专门提供只有当地人才知道的内部信息，
        包括小众景点、文化习俗和实用贴士。
        """

        recent_messages = state.get("messages", [])
        has_recent_search_result = False
        for msg in reversed(recent_messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue
            lowered = content.lower()
            if "搜索结果: [local_expert_skill]" in content or "搜索结果: [search_local_tips]" in content:
                has_recent_search_result = True
                break
            # 兼容历史未带工具标签的输出：local_expert_skill / RAG 结果通常包含 skill_route 或 source 引用标签
            if "搜索结果:" in content and (
                "skill_route:" in lowered
                or "[source=" in lowered
                or "本地建议" in content
                or "local_advice" in lowered
            ):
                has_recent_search_result = True
                break

        system_prompt = f"""您是当地专家智能体，专门从事内部知识和本地洞察。

您的专业领域包括：
- 当地习俗和文化细节
- 小众景点和小众推荐
- 本地餐饮和娱乐场所
- 实用的本地贴士和建议

当前规划请求：
- 目的地: {state.get('destination')}
- 兴趣: {', '.join(state.get('interests', []))}
- 时长: {state.get('duration')} 天

您的任务：提供当地专家洞察，包括：
1. 小众景点和当地人喜爱的地方
2. 文化礼仪和习俗
3. 本地餐饮推荐
4. 出行和省钱的内部贴士

工具策略：
- 您可以通过本地专家能力（含知识库/RAG）获取更可靠的本地信息
- 当需要检索时，请回复：'NEED_SEARCH: [本地知识库检索查询]'
- 获得工具结果后，请基于结果输出结论，避免编造信息
- 如结果中包含 [source=...#chunk=...] 引用标签，请尽量在结论中保留来源依据

执行约束：
- 当前是否已有可用检索结果：{"是" if has_recent_search_result else "否"}
- 若当前无检索结果，必须先回复：'NEED_SEARCH: [本地知识库检索查询]'
- 若当前已有检索结果，禁止再次发起 NEED_SEARCH，必须直接给出本地洞察结论

如果不需要检索，请直接提供您的本地专业知识和洞察。
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        
        response = self.llm.invoke(messages)
        
        # Store agent output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["local_expert"] = {
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "status": self._derive_agent_status(response.content)
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "local_expert"
        new_state["agent_outputs"] = agent_outputs
        
        return new_state
    
    def _itinerary_planner_agent(self, state: TravelPlanState) -> TravelPlanState:
        """
        行程规划师智能体，专门进行日程优化和物流安排

        这个智能体专门负责创建优化的日程安排，
        协调交通和活动的时间安排。
        """

        system_prompt = f"""您是行程规划师智能体，专门从事日程优化和物流安排。

您的专业领域包括：
- 每日行程规划和优化
- 交通和物流协调
- 时间管理和日程安排
- 活动排序和路线规划

当前规划请求：
- 目的地: {state.get('destination')}
- 时长: {state.get('duration')} 天
- 团队人数: {state.get('group_size')}
- 可用智能体洞察: {list(state.get('agent_outputs', {}).keys())}

您的任务：创建优化的行程安排，包括：
1. 逐日日程推荐
2. 活动的最佳时间安排
3. 地点间的交通建议
4. 休息时间和用餐安排

在创建行程时请考虑其他智能体的建议。
提供结构化的每日计划，最大化旅行体验。
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        
        response = self.llm.invoke(messages)
        
        # Store agent output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["itinerary_planner"] = {
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "status": self._derive_agent_status(response.content)
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "itinerary_planner"
        new_state["agent_outputs"] = agent_outputs
        
        return new_state
    
    def _tool_executor_node(self, state: TravelPlanState) -> TravelPlanState:
        """
        工具执行节点，根据智能体请求执行工具

        这个节点负责解析智能体的工具请求，
        并执行相应的搜索工具来获取实时信息。
        """

        last_message = state["messages"][-1] if state.get("messages") else None
        if not last_message:
            return state

        # 检查最后一条消息是否请求搜索
        content = last_message.content
        if "NEED_SEARCH:" in content:
            search_query = content.split("NEED_SEARCH:")[-1].strip()

            # 根据当前智能体和查询确定使用哪个工具
            current_agent = state.get("current_agent", "")
            agents_logger.info(f"[ToolExecutor] 解析到搜索需求 | 当前智能体: {current_agent} | 查询: {search_query}")
            
            try:
                # 智能工具选择：根据查询内容和当前智能体选择最合适的搜索工具
                selected_tool = ""
                if "weather" in search_query.lower() or "天气" in search_query or current_agent == "weather_analyst":
                    selected_tool = "search_weather_info"
                    # 天气相关查询：使用天气信息搜索工具
                    from tools.travel_tools import search_weather_info
                    tool_params = {"destination": state.get("destination", ""),
                                   "dates": state.get("travel_dates", "")}
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    # 该工具为异步工具，需使用 ainvoke 在独立事件循环中执行
                    import asyncio
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        tool_result = loop.run_until_complete(search_weather_info.ainvoke(tool_params))
                    finally:
                        loop.close()
                        try:
                            asyncio.set_event_loop(None)
                        except Exception:
                            pass
                elif current_agent == "local_expert":
                    selected_tool = "local_expert_skill"
                    # local_expert 固定优先走 skill（内部会路由 RAG / Search）
                    from tools.travel_tools import local_expert_skill
                    tool_params = {
                        "destination": state.get("destination", ""),
                        "interests": " ".join(state.get("interests", [])),
                        "query": search_query,
                        "top_k": 4
                    }
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    tool_result = local_expert_skill.invoke(tool_params)
                elif "attraction" in search_query.lower() or "activity" in search_query.lower() or "景点" in search_query or "活动" in search_query:
                    selected_tool = "search_attractions"
                    # 景点活动查询：使用景点搜索工具
                    from tools.travel_tools import search_attractions
                    tool_params = {"destination": state.get("destination", ""),
                                   "interests": " ".join(state.get("interests", []))}
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    tool_result = search_attractions.invoke(tool_params)
                elif "budget" in search_query.lower() or "cost" in search_query.lower() or "预算" in search_query or "费用" in search_query:
                    selected_tool = "search_budget_info"
                     # 预算费用查询：使用预算信息搜索工具
                    from tools.travel_tools import search_budget_info
                    tool_params = {"destination": state.get("destination", ""),
                                   "duration": str(state.get("duration", ""))}
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    tool_result = search_budget_info.invoke(tool_params)
                elif "hotel" in search_query.lower() or "accommodation" in search_query.lower() or "酒店" in search_query or "住宿" in search_query:
                    selected_tool = "search_hotels"
                    # 住宿查询：使用酒店搜索工具
                    from tools.travel_tools import search_hotels
                    tool_params = {"destination": state.get("destination", ""),
                                   "budget": state.get("budget_range", "mid-range")}
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    tool_result = search_hotels.invoke(tool_params)
                elif "restaurant" in search_query.lower() or "food" in search_query.lower() or "餐厅" in search_query or "美食" in search_query:
                    selected_tool = "search_restaurants"
                     # 餐饮查询：使用餐厅搜索工具
                    from tools.travel_tools import search_restaurants
                    tool_params = {"destination": state.get("destination", "")}
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    tool_result = search_restaurants.invoke(tool_params)
                elif "local" in search_query.lower() or "tip" in search_query.lower() or "本地" in search_query or "贴士" in search_query:
                    selected_tool = "local_expert_skill"
                    # 本地查询：优先使用本地专家能力（含RAG/搜索路由）
                    from tools.travel_tools import local_expert_skill
                    tool_params = {
                        "destination": state.get("destination", ""),
                        "interests": " ".join(state.get("interests", [])),
                        "query": search_query,
                        "top_k": 4
                    }
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    tool_result = local_expert_skill.invoke(tool_params)
                else:
                    selected_tool = "search_destination_info"
                    # 默认选择：使用目的地信息搜索工具
                    from tools.travel_tools import search_destination_info
                    tool_params = {"query": state.get("destination", "")}
                    agents_logger.info(f"[ToolExecutor] 调用工具: {selected_tool} | 参数: {tool_params}")
                    tool_result = search_destination_info.invoke(tool_params)

                # 记录工具返回结果大小（避免日志过大）
                result_str = str(tool_result)
                agents_logger.info(f"[ToolExecutor] 工具返回: {selected_tool} | 长度: {len(result_str)} 字符")

                # 将工具执行结果添加到消息历史中
                tool_message = AIMessage(content=f"搜索结果: [{selected_tool}] {tool_result}")
                new_state = state.copy()
                new_state["messages"] = state.get("messages", []) + [tool_message]
                return new_state

            except Exception as e:
                agents_logger.error(f"[ToolExecutor] 工具执行错误: {str(e)}")
                # 工具执行失败时添加错误消息
                error_message = AIMessage(content=f"工具执行错误: {str(e)}")
                new_state = state.copy()
                new_state["messages"] = state.get("messages", []) + [error_message]
                return new_state

        return state

    def _coordinator_router(self, state: TravelPlanState) -> str:
        """
        协调员路由器：从协调员决定下一步流程

        这个方法分析协调员的输出，决定下一步应该调用哪个智能体
        或执行哪个操作。这是LangGraph工作流的核心路由逻辑。

        参数：
        - state: 当前的旅行规划状态

        返回：下一个要执行的节点名称

        适用于大模型技术初级用户：
        这个路由器展示了如何在复杂的AI系统中实现智能决策，
        根据上下文动态选择下一步的执行路径。
        """

        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        if not last_message:
            agents_logger.info("[CoordinatorRouter] 无最近消息，结束流程")
            return "end"

        content = last_message.content.lower()
        missing_agents = self._get_missing_agents(state)

        # 路由决策逻辑：根据协调员的输出内容决定下一步行动
        agents_logger.info(f"[CoordinatorRouter] 协调员输出: {content}")

        # 检查协调员是否需要搜索工具
        if "search" in content or "need_search" in content or "搜索" in content:
            agents_logger.info("[CoordinatorRouter] 决策: 进入工具执行节点")
            return "tools"

        # 检查协调员是否请求特定的智能体
        if "travel_advisor" in content or "旅行顾问" in content:
            agents_logger.info("[CoordinatorRouter] 决策: 跳转 travel_advisor")
            return "travel_advisor"
        elif "weather_analyst" in content or "天气分析师" in content:
            agents_logger.info("[CoordinatorRouter] 决策: 跳转 weather_analyst")
            return "weather_analyst"
        elif "budget_optimizer" in content or "预算优化师" in content:
            agents_logger.info("[CoordinatorRouter] 决策: 跳转 budget_optimizer")
            return "budget_optimizer"
        elif "local_expert" in content or "当地专家" in content:
            agents_logger.info("[CoordinatorRouter] 决策: 跳转 local_expert")
            return "local_expert"
        elif "itinerary_planner" in content or "行程规划师" in content:
            agents_logger.info("[CoordinatorRouter] 决策: 跳转 itinerary_planner")
            return "itinerary_planner"
        elif "final_plan" in content or "最终计划" in content:
            if missing_agents:
                next_agent = missing_agents[0]
                agents_logger.warning(
                    f"[CoordinatorRouter] 检测到提前 FINAL_PLAN，但仍有未参与智能体: {missing_agents}，改为跳转 {next_agent}"
                )
                return next_agent
            agents_logger.info("[CoordinatorRouter] 决策: 结束流程")
            return "end"

        # 默认策略：检查哪些智能体还没有完成工作
        if missing_agents:
            next_agent = missing_agents[0]
            agents_logger.info(f"[CoordinatorRouter] 决策: 跳转 {next_agent} (待完成)")
            return next_agent

        # 如果所有智能体都已完成，结束流程
        agents_logger.info("[CoordinatorRouter] 决策: 所有智能体已完成，结束流程")
        return "end"
    
    def _agent_router(self, state: TravelPlanState) -> str:
        """
        智能体路由器：从智能体决定下一步流程

        这个方法处理各个专业智能体完成工作后的路由决策，
        决定是返回协调员还是执行工具搜索。

        参数：
        - state: 当前的旅行规划状态

        返回：下一个要执行的节点名称

        适用于大模型技术初级用户：
        这展示了多智能体系统中的反馈循环机制，
        智能体可以请求更多信息或将控制权交还给协调员。
        """

        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        if not last_message:
            agents_logger.info("[AgentRouter] 无最近消息，返回协调员")
            return "coordinator"

        content = last_message.content

        # 检查智能体是否需要搜索更多信息
        if "NEED_SEARCH:" in content:
            agents_logger.info("[AgentRouter] 检测到搜索需求，跳转工具节点")
            return "tools"

        # 否则返回协调员进行下一步决策
        agents_logger.info("[AgentRouter] 返回协调员继续决策")
        return "coordinator"
    
    def run_travel_planning(
        self,
        travel_request: Dict[str, Any],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的多智能体旅行规划工作流

        这是整个AI旅行规划智能体的主入口方法，
        它初始化状态、执行工作流并返回最终的旅行计划。

        参数：
        - travel_request: 包含旅行需求的字典

        返回：包含旅行计划和执行结果的字典

        适用于大模型技术初级用户：
        这个方法展示了如何将复杂的AI系统封装成简单的API，
        用户只需提供需求，系统就能自动协调多个智能体完成规划。
        """

        try:
            self._safe_event_emit(
                event_callback,
                event_type="session_initialized",
                message="会话已初始化，准备拆分并发任务。",
                progress=10,
                status="processing",
            )

            memory = self._build_short_term_memory(travel_request)
            subtasks = self._coordinator_skill_plan_subtasks(memory)
            memory["coordinator_plan"] = subtasks

            self._safe_event_emit(
                event_callback,
                event_type="coordinator_planned",
                message="协调员已完成任务拆分，开始并发分析阶段。",
                progress=20,
                agent="coordinator",
                status="processing",
                data={"subtasks": subtasks},
            )

            analysis_agents = self._analysis_agents()
            completed_count = 0
            analysis_total = len(analysis_agents)

            with ThreadPoolExecutor(max_workers=analysis_total) as executor:
                future_map = {}
                for agent in analysis_agents:
                    memory["agent_slots"][agent]["subtask"] = subtasks.get(agent, "")
                    self._safe_event_emit(
                        event_callback,
                        event_type="agent_started",
                        message=f"{agent} 已启动并发分析。",
                        agent=agent,
                        progress=25,
                        status="processing",
                    )
                    future = executor.submit(
                        self._run_analysis_agent_with_private_context,
                        agent,
                        subtasks.get(agent, ""),
                        memory,
                        event_callback,
                    )
                    future_map[future] = agent

                for future in as_completed(future_map):
                    agent = future_map[future]
                    try:
                        slot_result = future.result()
                    except Exception as future_err:
                        slot_result = {
                            "status": "failed",
                            "subtask": subtasks.get(agent, ""),
                            "agent_messages": [],
                            "tool_artifacts": [],
                            "output": "",
                            "error": str(future_err),
                            "started_at": "",
                            "finished_at": datetime.now().isoformat(),
                        }

                    memory["agent_slots"][agent] = slot_result
                    memory["tool_artifacts"].extend(slot_result.get("tool_artifacts", []))

                    completed_count += 1
                    progress = 20 + int((completed_count / analysis_total) * 45)
                    if slot_result.get("status") == "completed":
                        self._safe_event_emit(
                            event_callback,
                            event_type="agent_completed",
                            message=f"{agent} 分析完成。",
                            agent=agent,
                            progress=progress,
                            status="processing",
                        )
                    else:
                        self._safe_event_emit(
                            event_callback,
                            event_type="agent_failed",
                            message=f"{agent} 执行失败: {slot_result.get('error', '未知错误')}",
                            agent=agent,
                            progress=progress,
                            status="processing",
                        )

            self._safe_event_emit(
                event_callback,
                event_type="collector_started",
                message="进入汇总阶段，正在归一化并发结果。",
                progress=70,
                status="processing",
            )
            collector_output = self._collector_stage(memory)

            self._safe_event_emit(
                event_callback,
                event_type="collector_completed",
                message="汇总完成，开始行程整合。",
                progress=78,
                status="processing",
                data={
                    "missing_agents": collector_output.get("missing_agents", []),
                    "failed_agents": collector_output.get("failed_agents", []),
                },
            )

            itinerary_text = self._itinerary_planner_stage(memory, collector_output)
            memory["agent_slots"]["itinerary_planner"] = {
                "status": "completed" if itinerary_text else "failed",
                "subtask": subtasks.get("itinerary_planner", ""),
                "agent_messages": [],
                "tool_artifacts": [],
                "output": itinerary_text,
                "error": "" if itinerary_text else "itinerary_planner 未生成内容",
                "started_at": datetime.now().isoformat(),
                "finished_at": datetime.now().isoformat(),
            }

            self._safe_event_emit(
                event_callback,
                event_type="itinerary_completed",
                message="行程规划师已完成串行整合。",
                progress=90,
                agent="itinerary_planner",
                status="processing",
            )

            self._safe_event_emit(
                event_callback,
                event_type="coordinator_finalizing",
                message="协调员正在审阅行程并生成最终结论。",
                progress=94,
                agent="coordinator",
                status="processing",
            )
            coordinator_output = self._coordinator_finalize_stage(
                memory,
                collector_output,
                itinerary_text,
            )
            self._safe_event_emit(
                event_callback,
                event_type="coordinator_finalized",
                message="协调员已完成终审，准备输出最终方案。",
                progress=97,
                agent="coordinator",
                status="processing",
            )

            final_markdown = self._final_summarizer_stage(
                memory,
                collector_output,
                itinerary_text,
                coordinator_output,
            )
            missing_agents = collector_output.get("missing_agents", [])

            agent_outputs: Dict[str, Any] = {}
            for agent_name, slot in memory.get("agent_slots", {}).items():
                agent_outputs[agent_name] = {
                    "response": slot.get("output", ""),
                    "timestamp": slot.get("finished_at", ""),
                    "status": slot.get("status", ""),
                    "tool_count": len(slot.get("tool_artifacts", []) or []),
                    "error": slot.get("error", ""),
                }

            travel_plan = {
                "destination": memory["shared_facts"].get("destination"),
                "duration": memory["shared_facts"].get("duration"),
                "travel_dates": memory["shared_facts"].get("travel_dates"),
                "group_size": memory["shared_facts"].get("group_size"),
                "budget_range": memory["shared_facts"].get("budget_range"),
                "interests": memory["shared_facts"].get("interests"),
                "planning_method": "共享会话并发多Agent架构",
                "summary": "并发分析 + 串行整合生成的旅行方案",
                "agent_contributions": {
                    "travel_advisor": memory["agent_slots"]["travel_advisor"].get("output", ""),
                    "weather_analyst": memory["agent_slots"]["weather_analyst"].get("output", ""),
                    "budget_optimizer": memory["agent_slots"]["budget_optimizer"].get("output", ""),
                    "local_expert": memory["agent_slots"]["local_expert"].get("output", ""),
                    "itinerary_planner": itinerary_text,
                    "coordinator": coordinator_output,
                },
                "recommendations": {
                    "destination_highlights": "参见旅行顾问输出",
                    "weather_considerations": "参见天气分析师输出",
                    "budget_breakdown": "参见预算优化师输出",
                    "local_insights": "参见当地专家输出（skill + RAG）",
                    "daily_itinerary": "参见行程规划师输出",
                },
                "missing_agents": missing_agents,
                "coordinator_finalization": coordinator_output,
                "final_plan": final_markdown,
            }

            self._safe_event_emit(
                event_callback,
                event_type="task_completed",
                message="并发规划完成。",
                progress=100,
                status="completed",
                agent="coordinator",
                data={"missing_agents": missing_agents},
            )

            agent_status_snapshot = self._build_agent_slot_status_snapshot(memory)
            return {
                "success": True,
                "travel_plan": travel_plan,
                "agent_outputs": agent_outputs,
                "total_iterations": 1,
                "planning_complete": len(missing_agents) == 0,
                "missing_agents": missing_agents,
                "short_term_memory": {
                    "session_id": memory.get("session_id"),
                    "shared_facts": memory.get("shared_facts", {}),
                    "merge_notes": memory.get("merge_notes", []),
                    "timeline": memory.get("timeline", []),
                    "coordinator_plan": memory.get("coordinator_plan", {}),
                    "collector_output": memory.get("collector_output", {}),
                    "itinerary_output": memory.get("itinerary_output", ""),
                    "coordinator_final_output": memory.get("coordinator_final_output", ""),
                    "agent_slots": agent_status_snapshot,
                },
            }

        except Exception as e:
            self._safe_event_emit(
                event_callback,
                event_type="task_failed",
                message=f"规划失败: {str(e)}",
                progress=100,
                status="failed",
            )
            return {
                "success": False,
                "error": f"规划过程中出现错误: {str(e)}",
                "travel_plan": {},
                "agent_outputs": {},
                "total_iterations": 0,
                "planning_complete": False,
            }
    
    def _compile_final_plan(self, state: TravelPlanState) -> Dict[str, Any]:
        """
        从所有智能体输出编译最终旅行计划

        这个方法整合所有专业智能体的建议和分析，
        生成一个完整、结构化的旅行计划。

        参数：
        - state: 包含所有智能体输出的最终状态

        返回：完整的旅行计划字典

        适用于大模型技术初级用户：
        这个方法展示了如何将多个AI智能体的输出
        整合成一个统一、有用的最终产品。
        """

        agent_outputs = state.get("agent_outputs", {})
        missing_agents = self._get_missing_agents(state)

        # 构建基础旅行计划结构
        final_plan = {
            "destination": state.get("destination"),                      # 目的地
            "duration": state.get("duration"),                           # 旅行时长
            "travel_dates": state.get("travel_dates"),                   # 旅行日期
            "group_size": state.get("group_size"),                       # 团队人数
            "budget_range": state.get("budget_range"),                   # 预算范围
            "interests": state.get("interests"),                         # 兴趣爱好
            "planning_method": "LangGraph多智能体协作",                   # 规划方法
            "agent_contributions": {},                                    # 智能体贡献
            "recommendations": {},                                        # 推荐建议
            "summary": "使用LangGraph框架的多智能体协作生成的旅行计划",     # 计划摘要
            "missing_agents": missing_agents                               # 尚未完成的智能体
        }

        # 从每个智能体提取关键信息
        for agent_name, output in agent_outputs.items():
            agent_name_cn = {
                'travel_advisor': '旅行顾问',
                'weather_analyst': '天气分析师',
                'budget_optimizer': '预算优化师',
                'local_expert': '当地专家',
                'itinerary_planner': '行程规划师'
            }.get(agent_name, agent_name)

            final_plan["agent_contributions"][agent_name_cn] = {
                "contribution": output.get("response", ""),               # 智能体的具体建议
                "timestamp": output.get("timestamp", ""),                 # 生成时间戳
                "status": output.get("status", "")                       # 执行状态
            }

        # 生成总结性推荐
        if agent_outputs:
            final_plan["recommendations"] = {
                "destination_highlights": "查看旅行顾问推荐",
                "weather_considerations": "查看天气分析师洞察",
                "budget_breakdown": "查看预算优化师分析",
                "local_insights": "遵循当地专家贴士",
                "daily_itinerary": "使用行程规划师日程"
            }

        if missing_agents:
            final_plan["summary"] = (
                "多智能体已部分完成；以下智能体尚未完成最终可用输出: "
                + ", ".join(missing_agents)
            )

        # 生成可直接落盘的最终正文，优先使用行程规划师输出
        itinerary_output = agent_outputs.get("itinerary_planner", {})
        itinerary_text = itinerary_output.get("response", "") if isinstance(itinerary_output, dict) else ""
        if isinstance(itinerary_text, str) and itinerary_text.strip() and "NEED_SEARCH:" not in itinerary_text:
            final_plan["final_plan"] = itinerary_text.strip()
        else:
            candidate_texts: List[str] = []
            for output in agent_outputs.values():
                if not isinstance(output, dict):
                    continue
                text = output.get("response", "")
                status = str(output.get("status", "")).lower()
                if isinstance(text, str) and text.strip() and status == "completed" and "NEED_SEARCH:" not in text:
                    candidate_texts.append(text.strip())
            if candidate_texts:
                candidate_texts.sort(key=len, reverse=True)
                final_plan["final_plan"] = candidate_texts[0]

        return final_plan

