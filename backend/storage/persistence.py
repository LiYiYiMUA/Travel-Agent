from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from redis import Redis
from redis.exceptions import RedisError
import psycopg
from psycopg.types.json import Json


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int
    password: str
    task_ttl_seconds: int


class RedisStateStore:
    """
    Redis-backed state store for:
    - task meta/status
    - request/result snapshots
    - short-term memory
    - event stream (for streaming/debug)
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("redis_state_store")
        self.cfg = RedisConfig(
            host=os.getenv("REDIS_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("REDIS_PORT", "6379").strip()),
            db=int(os.getenv("REDIS_DB", "0").strip()),
            password=os.getenv("REDIS_PASSWORD", "").strip(),
            task_ttl_seconds=int(os.getenv("REDIS_TASK_TTL_SECONDS", "604800").strip()),
        )
        self._client: Optional[Redis] = None
        self.enabled: bool = False
        self._connect()

    def _connect(self) -> None:
        try:
            self._client = Redis(
                host=self.cfg.host,
                port=self.cfg.port,
                db=self.cfg.db,
                password=self.cfg.password or None,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            self._client.ping()
            self.enabled = True
            self.logger.info(
                f"[RedisStateStore] connected: {self.cfg.host}:{self.cfg.port}/{self.cfg.db}"
            )
        except Exception as exc:
            self._client = None
            self.enabled = False
            self.logger.warning(f"[RedisStateStore] disabled: {exc}")

    @staticmethod
    def _meta_key(task_id: str) -> str:
        return f"tp:task:{task_id}:meta"

    @staticmethod
    def _request_key(task_id: str) -> str:
        return f"tp:task:{task_id}:request"

    @staticmethod
    def _result_key(task_id: str) -> str:
        return f"tp:task:{task_id}:result"

    @staticmethod
    def _memory_key(task_id: str) -> str:
        return f"tp:task:{task_id}:short_term_memory"

    @staticmethod
    def _events_key(task_id: str) -> str:
        return f"tp:task:{task_id}:events"

    def _expire_all(self, task_id: str) -> None:
        if not self.enabled or not self._client:
            return
        ttl = self.cfg.task_ttl_seconds
        self._client.expire(self._meta_key(task_id), ttl)
        self._client.expire(self._request_key(task_id), ttl)
        self._client.expire(self._result_key(task_id), ttl)
        self._client.expire(self._memory_key(task_id), ttl)
        self._client.expire(self._events_key(task_id), ttl)

    def upsert_task(self, task_id: str, task: Dict[str, Any]) -> None:
        if not self.enabled or not self._client:
            return
        try:
            meta = {
                "task_id": task_id,
                "status": str(task.get("status", "")),
                "progress": str(task.get("progress", 0)),
                "current_agent": str(task.get("current_agent", "")),
                "message": str(task.get("message", "")),
                "created_at": str(task.get("created_at", "")),
                "updated_at": str(task.get("updated_at", "")),
                "result_file": str(task.get("result_file", "")),
                "result_markdown_file": str(task.get("result_markdown_file", "")),
            }
            self._client.hset(self._meta_key(task_id), mapping=meta)

            if task.get("request") is not None:
                self._client.set(self._request_key(task_id), _json_dumps(task.get("request")))

            if task.get("result") is not None:
                self._client.set(self._result_key(task_id), _json_dumps(task.get("result")))

            self._expire_all(task_id)
        except RedisError as exc:
            self.logger.warning(f"[RedisStateStore] upsert_task failed task={task_id}: {exc}")

    def append_event(self, task_id: str, event: Dict[str, Any]) -> None:
        if not self.enabled or not self._client:
            return
        try:
            payload = {
                "seq": str(event.get("seq", "")),
                "type": str(event.get("type", "")),
                "message": str(event.get("message", "")),
                "timestamp": str(event.get("timestamp", "")),
                "progress": str(event.get("progress", "")),
                "agent": str(event.get("agent", "")),
                "status": str(event.get("status", "")),
                "data_json": _json_dumps(event.get("data", {})),
            }
            self._client.xadd(self._events_key(task_id), payload)
            self._client.expire(self._events_key(task_id), self.cfg.task_ttl_seconds)
        except RedisError as exc:
            self.logger.warning(f"[RedisStateStore] append_event failed task={task_id}: {exc}")

    def save_short_term_memory(self, task_id: str, memory: Dict[str, Any]) -> None:
        if not self.enabled or not self._client:
            return
        try:
            self._client.set(self._memory_key(task_id), _json_dumps(memory))
            self._client.expire(self._memory_key(task_id), self.cfg.task_ttl_seconds)
            self.logger.info(f"[RedisStateStore] short_term_memory saved task={task_id}")
        except RedisError as exc:
            self.logger.warning(
                f"[RedisStateStore] save_short_term_memory failed task={task_id}: {exc}"
            )

    def get_task_snapshot(self, task_id: str) -> Dict[str, Any]:
        if not self.enabled or not self._client:
            return {}
        try:
            meta = self._client.hgetall(self._meta_key(task_id))
            if not meta:
                return {}
            request_json = self._client.get(self._request_key(task_id))
            result_json = self._client.get(self._result_key(task_id))
            memory_json = self._client.get(self._memory_key(task_id))
            events = self._client.xrange(self._events_key(task_id), min="-", max="+", count=200)
            event_items: List[Dict[str, Any]] = []
            for _, fields in events:
                item = dict(fields)
                item["data"] = _json_loads(item.get("data_json"), {})
                item.pop("data_json", None)
                event_items.append(item)
            return {
                "meta": meta,
                "request": _json_loads(request_json, {}),
                "result": _json_loads(result_json, {}),
                "short_term_memory": _json_loads(memory_json, {}),
                "events": event_items,
            }
        except RedisError as exc:
            self.logger.warning(f"[RedisStateStore] get_task_snapshot failed task={task_id}: {exc}")
            return {}


@dataclass
class PostgresConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str


class PostgresResultStore:
    """PostgreSQL-backed store for final planning result snapshots."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("postgres_result_store")
        self.cfg = PostgresConfig(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("POSTGRES_PORT", "5432").strip()),
            user=os.getenv("POSTGRES_USER", "").strip(),
            password=os.getenv("POSTGRES_PASSWORD", "").strip(),
            dbname=os.getenv("POSTGRES_DB", "postgres").strip(),
        )
        self.enabled = all([self.cfg.host, self.cfg.port, self.cfg.user, self.cfg.dbname])
        if self.enabled:
            self._ensure_table()
        else:
            self.logger.info("[PostgresResultStore] disabled: missing postgres env config")

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            user=self.cfg.user,
            password=self.cfg.password,
            dbname=self.cfg.dbname,
            connect_timeout=5,
        )

    def _ensure_table(self) -> None:
        if not self.enabled:
            return
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS travel_planning_results (
                            id BIGSERIAL PRIMARY KEY,
                            task_id TEXT UNIQUE NOT NULL,
                            status TEXT NOT NULL,
                            destination TEXT,
                            request_json JSONB NOT NULL,
                            result_json JSONB NOT NULL,
                            short_term_memory_json JSONB,
                            final_plan_markdown TEXT,
                            agent_participation_json JSONB,
                            planning_complete BOOLEAN DEFAULT FALSE,
                            missing_agents JSONB,
                            result_file TEXT,
                            result_markdown_file TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_travel_planning_results_task_id "
                        "ON travel_planning_results(task_id)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_travel_planning_results_created_at "
                        "ON travel_planning_results(created_at DESC)"
                    )
                    # Backward-compatible schema migration for existing deployments.
                    cur.execute(
                        "ALTER TABLE travel_planning_results "
                        "ADD COLUMN IF NOT EXISTS agent_participation_json JSONB"
                    )
            self.logger.info(
                f"[PostgresResultStore] connected and ensured table on {self.cfg.host}:{self.cfg.port}/{self.cfg.dbname}"
            )
        except Exception as exc:
            self.enabled = False
            self.logger.warning(f"[PostgresResultStore] disabled: {exc}")

    def upsert_result(
        self,
        task_id: str,
        request: Dict[str, Any],
        result: Dict[str, Any],
        *,
        status: str,
        result_file: str,
        result_markdown_file: str,
        final_plan_markdown: str = "",
        missing_agents_override: Optional[List[str]] = None,
        agent_participation: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        travel_plan = result.get("travel_plan", {}) if isinstance(result, dict) else {}
        destination = str(travel_plan.get("destination", request.get("destination", "")))
        short_term_memory = result.get("short_term_memory", {})
        final_plan_text = str(final_plan_markdown or travel_plan.get("final_plan", "")).strip()
        planning_complete = bool(result.get("planning_complete", False))
        missing_agents = (
            missing_agents_override
            if isinstance(missing_agents_override, list)
            else result.get("missing_agents", [])
        )
        participation = (
            agent_participation
            if isinstance(agent_participation, dict)
            else result.get("agent_participation", {})
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO travel_planning_results (
                            task_id, status, destination, request_json, result_json,
                            short_term_memory_json, final_plan_markdown, agent_participation_json, planning_complete,
                            missing_agents, result_file, result_markdown_file, created_at, updated_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                        )
                        ON CONFLICT (task_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            destination = EXCLUDED.destination,
                            request_json = EXCLUDED.request_json,
                            result_json = EXCLUDED.result_json,
                            short_term_memory_json = EXCLUDED.short_term_memory_json,
                            final_plan_markdown = EXCLUDED.final_plan_markdown,
                            agent_participation_json = EXCLUDED.agent_participation_json,
                            planning_complete = EXCLUDED.planning_complete,
                            missing_agents = EXCLUDED.missing_agents,
                            result_file = EXCLUDED.result_file,
                            result_markdown_file = EXCLUDED.result_markdown_file,
                            updated_at = NOW()
                        """,
                        (
                            task_id,
                            status,
                            destination,
                            Json(request),
                            Json(result),
                            Json(short_term_memory if isinstance(short_term_memory, dict) else {}),
                            final_plan_text,
                            Json(participation if isinstance(participation, dict) else {}),
                            planning_complete,
                            Json(missing_agents if isinstance(missing_agents, list) else []),
                            result_file,
                            result_markdown_file,
                        ),
                    )
            self.logger.info(f"[PostgresResultStore] result upserted task={task_id}")
        except Exception as exc:
            self.logger.warning(f"[PostgresResultStore] upsert_result failed task={task_id}: {exc}")

    def get_result(self, task_id: str) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT task_id, status, destination, request_json, result_json,
                               short_term_memory_json, final_plan_markdown, agent_participation_json,
                               planning_complete, missing_agents,
                               result_file, result_markdown_file, created_at, updated_at
                        FROM travel_planning_results
                        WHERE task_id = %s
                        """,
                        (task_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return {}
            return {
                "task_id": row[0],
                "status": row[1],
                "destination": row[2],
                "request_json": row[3],
                "result_json": row[4],
                "short_term_memory_json": row[5],
                "final_plan_markdown": row[6],
                "agent_participation_json": row[7],
                "planning_complete": row[8],
                "missing_agents": row[9],
                "result_file": row[10],
                "result_markdown_file": row[11],
                "created_at": str(row[12]),
                "updated_at": str(row[13]),
            }
        except Exception as exc:
            self.logger.warning(f"[PostgresResultStore] get_result failed task={task_id}: {exc}")
            return {}
