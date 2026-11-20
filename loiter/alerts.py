"""Alert sinks. The engine returns events; the caller decides where they go."""
import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class JsonlAlertSink:
    """Appends one JSON object per line. Creates parent directories as needed."""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def emit(self, event: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        logger.info("alert %s person=%s zone=%s dwell=%ss", event["camera_id"], event["person_id"],
                    event["zone"], event["dwell_time_seconds"])


class ListAlertSink:
    """In-memory sink for tests and batch runs."""

    def __init__(self) -> None:
        self.events: List[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)
