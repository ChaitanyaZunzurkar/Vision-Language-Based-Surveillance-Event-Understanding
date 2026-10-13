"""Validation for precomputed temporal-window JSON inputs."""

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Dict, List


def validate_windows_payload(
    payload: Any,
    expected_video_names: set[str],
    duration_sec: float,
) -> List[Dict[str, Any]]:
    """Validate the notebooks' {"windows": [...]} schema and return a safe copy."""
    if not isinstance(payload, dict) or not isinstance(payload.get("windows"), list):
        raise ValueError("Expected a JSON object with a top-level 'windows' array")
    if not payload["windows"]:
        raise ValueError("The windows array must contain at least one temporal window")

    windows = deepcopy(payload["windows"])
    if any(
        not isinstance(window, dict)
        or not isinstance(window.get("video_id"), str)
        or not window["video_id"].strip()
        for window in windows
    ):
        raise ValueError("Every temporal window must have a non-empty video_id")
    source_ids = {
        str(window["video_id"])
        for window in windows
    }
    if source_ids and not source_ids.issubset(expected_video_names):
        raise ValueError(
            f"JSON video ID(s) {sorted(source_ids)} do not match the uploaded video "
            f"({sorted(expected_video_names)})"
        )

    for window_index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"Window {window_index} must be an object")
        time_range = window.get("window")
        if (
            not isinstance(time_range, list)
            or len(time_range) != 2
            or not all(_is_finite_number(value) for value in time_range)
        ):
            raise ValueError(f"Window {window_index} must have a numeric [start, end] range")
        window_start, window_end = map(float, time_range)
        if window_start < 0 or window_end <= window_start:
            raise ValueError(f"Window {window_index} has an invalid time range")
        if window_start > duration_sec + 1:
            raise ValueError(f"Window {window_index} starts after the uploaded video ends")
        if not isinstance(window.get("tracks"), list):
            raise ValueError(f"Window {window_index} must contain a tracks array")

        known_track_ids = set()
        for track_index, track in enumerate(window["tracks"]):
            if not isinstance(track, dict):
                raise ValueError(f"Window {window_index} track {track_index} must be an object")
            track_id = track.get("track_id")
            if isinstance(track_id, bool) or not isinstance(track_id, int):
                raise ValueError(f"Window {window_index} track {track_index} needs an integer track_id")
            if not isinstance(track.get("class"), str) or not track["class"].strip():
                raise ValueError(f"Window {window_index} track {track_index} needs a class")
            known_track_ids.add(track_id)

        events = window.get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"Window {window_index} events must be an array")
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"Window {window_index} event {event_index} must be an object")
            if not isinstance(event.get("event_type"), str) or not event["event_type"].strip():
                raise ValueError(f"Window {window_index} event {event_index} needs an event_type")
            if not isinstance(event.get("description"), str) or not event["description"].strip():
                raise ValueError(f"Window {window_index} event {event_index} needs a description")
            start = event.get("abs_start", event.get("start"))
            end = event.get("abs_end", event.get("end"))
            if not _is_finite_number(start) or not _is_finite_number(end):
                raise ValueError(f"Window {window_index} event {event_index} needs numeric times")
            start, end = float(start), float(end)
            if ("abs_start" in event) != ("abs_end" in event):
                raise ValueError(
                    f"Window {window_index} event {event_index} must provide both absolute timestamps"
                )
            if end < start:
                raise ValueError(f"Window {window_index} event {event_index} has an invalid time range")
            confidence = event.get("confidence", 0.8)
            if not _is_finite_number(confidence) or not 0 <= float(confidence) <= 1:
                raise ValueError(
                    f"Window {window_index} event {event_index} confidence must be between 0 and 1"
                )
            if "abs_start" in event:
                if start < 0 or end > duration_sec + 1:
                    raise ValueError(
                        f"Window {window_index} event {event_index} absolute times exceed the video"
                    )
                if "start" in event and "end" in event:
                    relative_start, relative_end = event["start"], event["end"]
                    if (
                        not _is_finite_number(relative_start)
                        or not _is_finite_number(relative_end)
                        or abs(float(relative_start) + window_start - start) > 1
                        or abs(float(relative_end) + window_start - end) > 1
                    ):
                        raise ValueError(
                            f"Window {window_index} event {event_index} relative and absolute timestamps disagree"
                        )
            elif start < 0 or end > window_end - window_start + 1:
                raise ValueError(
                    f"Window {window_index} event {event_index} relative times exceed its window"
                )
            for field in ("subject_track", "object_track"):
                track_id = event.get(field)
                if track_id is not None and (
                    isinstance(track_id, bool)
                    or not isinstance(track_id, int)
                    or track_id not in known_track_ids
                ):
                    raise ValueError(
                        f"Window {window_index} event {event_index} references an unknown {field}"
                    )

    return windows


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def matching_video_names(filename: str, video_id: str) -> set[str]:
    """Names accepted in notebook window files for an ingested video."""
    return {video_id, Path(filename).stem}


def clip_event_times_to_video(payload: Any, duration_sec: float) -> Any:
    """Clip precomputed event timestamps to the source video's valid interval."""
    if not isinstance(payload, dict) or not isinstance(payload.get("windows"), list):
        return payload

    clipped_payload = deepcopy(payload)
    for window in clipped_payload["windows"]:
        if not isinstance(window, dict):
            continue
        time_range = window.get("window")
        if not isinstance(time_range, list) or len(time_range) != 2:
            continue
        window_start = time_range[0]
        if not _is_finite_number(window_start):
            continue

        events = window.get("events")
        if not isinstance(events, list):
            continue

        retained_events = []
        for event in events:
            if not isinstance(event, dict):
                retained_events.append(event)
                continue
            start = event.get("abs_start", event.get("start"))
            end = event.get("abs_end", event.get("end"))
            if not _is_finite_number(start) or not _is_finite_number(end):
                retained_events.append(event)
                continue

            absolute_start = float(start)
            absolute_end = float(end)
            if "abs_start" not in event:
                absolute_start += float(window_start)
                absolute_end += float(window_start)
            if absolute_start >= duration_sec:
                continue

            clipped_start = max(0.0, absolute_start)
            clipped_end = min(duration_sec, absolute_end)
            if clipped_end < clipped_start:
                continue
            if "abs_start" in event:
                event["abs_start"] = clipped_start
                event["abs_end"] = clipped_end
            event["start"] = max(0.0, clipped_start - float(window_start))
            event["end"] = max(0.0, clipped_end - float(window_start))
            retained_events.append(event)

        window["events"] = retained_events

    return clipped_payload
