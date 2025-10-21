# src/eval/langsmith_eval.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional


def _get_client() -> Optional["Client"]:
    """Safely initialize LangSmith client if configured; never raise."""
    if not os.getenv("LANGCHAIN_API_KEY"):
        return None
    try:
        from langsmith import Client  # type: ignore
        return Client()
    except Exception:
        return None


def _get_or_create_dataset_id(client, dataset_name: str) -> Optional[str]:
    """Return dataset_id; create dataset if it doesn't exist. Never raise."""
    if client is None:
        return None

    # Try new SDK method
    try:
        ds = client.read_dataset(dataset_name=dataset_name)  # type: ignore[attr-defined]
        ds_id = getattr(ds, "id", None)
        if ds_id:
            return ds_id
    except Exception:
        pass

    # Fallback: list and match
    try:
        for ds in client.list_datasets():  # type: ignore[attr-defined]
            if getattr(ds, "name", None) == dataset_name:
                ds_id = getattr(ds, "id", None)
                if ds_id:
                    return ds_id
    except Exception:
        pass

    # Create new dataset
    try:
        ds = client.create_dataset(  # type: ignore[attr-defined]
            dataset_name=dataset_name,
            description="Auto-created by LangGraph RAG + Weather app",
        )
        return getattr(ds, "id", None)
    except Exception:
        return None


def record_eval(
    example_input: Dict[str, Any],
    model_output: Dict[str, Any],
    run_name: str = "agent-run"
) -> None:
    """
    Best-effort LangSmith logging.
    - Controlled by LANGSMITH_LOG_EXAMPLES=1 (default off).
    - Uses LANGCHAIN_PROJECT as dataset name (defaults to 'ai-pipeline-assignment').
    - Absolutely never raises exceptions.
    """
    if os.getenv("LANGSMITH_LOG_EXAMPLES", "0") != "1":
        return

    client = _get_client()
    if client is None:
        return

    dataset_name = os.getenv("LANGCHAIN_PROJECT", "ai-pipeline-assignment")
    try:
        ds_id = _get_or_create_dataset_id(client, dataset_name)
        if not ds_id:
            return

        # Write a single example row into the dataset
        client.create_example(  # type: ignore[attr-defined]
            inputs=example_input,
            outputs=model_output,
            dataset_id=ds_id,
            created_by=run_name,
        )
    except Exception:
        # Telemetry must never break UX
        return
