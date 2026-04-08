import os
import sys
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from storage.persistence import PostgresResultStore, RedisStateStore


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


def test_redis_state_store_roundtrip():
    _load_env_file()
    store = RedisStateStore()
    assert store.enabled, "Redis should be available for integration test"

    task_id = f"pytest-{uuid.uuid4()}"
    task = {
        "status": "processing",
        "progress": 42,
        "current_agent": "coordinator",
        "message": "running",
        "request": {"destination": "深圳"},
        "result": {"ok": True},
    }
    store.upsert_task(task_id, task)
    store.append_event(
        task_id,
        {
            "seq": 1,
            "type": "task_update",
            "message": "running",
            "status": "processing",
            "progress": 42,
        },
    )
    store.save_short_term_memory(task_id, {"facts": {"destination": "深圳"}})

    snapshot = store.get_task_snapshot(task_id)
    assert snapshot.get("meta", {}).get("status") == "processing"
    assert snapshot.get("request", {}).get("destination") == "深圳"
    assert snapshot.get("short_term_memory", {}).get("facts", {}).get("destination") == "深圳"
    assert len(snapshot.get("events", [])) >= 1


def test_postgres_result_store_upsert_and_get():
    _load_env_file()
    store = PostgresResultStore()
    assert store.enabled, "PostgreSQL should be available for integration test"

    task_id = f"pytest-{uuid.uuid4()}"
    request = {"destination": "郑州", "group_size": 2}
    result = {
        "travel_plan": {
            "destination": "郑州",
            "final_plan": "test markdown plan",
        },
        "short_term_memory": {"facts": {"destination": "郑州"}},
        "planning_complete": True,
        "missing_agents": [],
    }

    store.upsert_result(
        task_id,
        request,
        result,
        status="completed",
        result_file="pytest_result.json",
        result_markdown_file="pytest_result.md",
        final_plan_markdown="# Final Report\nDay 1",
        missing_agents_override=[],
        agent_participation={
            "expected_agents": ["travel_advisor"],
            "participated_agents": ["travel_advisor"],
            "not_participating_agents": [],
            "all_expected_participated": True,
        },
    )

    row = store.get_result(task_id)
    assert row.get("task_id") == task_id
    assert row.get("status") == "completed"
    assert row.get("destination") == "郑州"
    assert row.get("planning_complete") is True
    assert (row.get("final_plan_markdown") or "").startswith("# Final Report")
    assert row.get("agent_participation_json", {}).get("all_expected_participated") is True
