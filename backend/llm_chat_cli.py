#!/usr/bin/env python3
"""Minimal CLI chat script for the project's configured OpenAI-compatible LLM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def load_env() -> None:
    backend_dir = Path(__file__).resolve().parent
    env_path = backend_dir / ".env"
    load_dotenv(env_path)


def build_llm(temperature: float) -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    missing = []
    if not model:
        missing.append("OPENAI_MODEL")
    if not base_url:
        missing.append("OPENAI_BASE_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the configured LLM using backend/.env settings."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Single-turn prompt. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt to use for the conversation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print model/base_url and whether API key is configured.",
    )
    return parser.parse_args()


def print_config() -> None:
    model = os.getenv("OPENAI_MODEL", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    print(f"model={model}")
    print(f"base_url={base_url}")
    print(f"api_key_configured={has_key}")


def ask_once(llm: ChatOpenAI, history: list, user_text: str) -> str:
    history.append(HumanMessage(content=user_text))
    response = llm.invoke(history)
    text = str(response.content)
    history.append(AIMessage(content=text))
    return text


def run_single_turn(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> int:
    history = [SystemMessage(content=system_prompt)]
    answer = ask_once(llm, history, user_prompt)
    print(answer)
    return 0


def run_interactive(llm: ChatOpenAI, system_prompt: str) -> int:
    print("Interactive mode. Type /exit to quit.")
    history = [SystemMessage(content=system_prompt)]

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            break

        try:
            answer = ask_once(llm, history, user_text)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"LLM> {answer}")

    return 0


def main() -> int:
    args = parse_args()
    load_env()

    if args.show_config:
        print_config()

    try:
        llm = build_llm(args.temperature)
    except Exception as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 1

    try:
        if args.prompt:
            return run_single_turn(llm, args.system, args.prompt)
        return run_interactive(llm, args.system)
    except Exception as exc:
        print(f"Request error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
