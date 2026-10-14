"""Qwen2.5-VL temporal-window event extraction following the Stage 4 notebooks."""

import json
import math
import os
import re
import sys
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from server.src.config.loader import config_loader
from server.src.utils.device import DEVICE_INFO
from server.src.utils.logger import logger


SYSTEM_PROMPT = (
    "You describe video surveillance events. Reply ONLY with JSON matching the given schema, "
    "no preamble, no markdown fences. Use subject_track and object_track values from the "
    "'Tracked objects' list below. Confirm candidate events only when the video supports them. "
    "If no specific action occurs, return a scene_description event. Always return at least one "
    "descriptive event. 'start' and 'end' are seconds measured from the START of the window."
)
SCHEMA_HINT = (
    '{"events":[{"event_type":str,"subject_track":int_or_null,"object_track":int_or_null,'
    '"start":float,"end":float,"zone":str_or_null,"confidence":float,"description":str}]}'
)


class VLMEventUnderstanding:
    """Loads Qwen lazily and verifies its output against the active window's track IDs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.model = None
        self.processor = None
        self.execution_device = DEVICE_INFO.device if device == "auto" else device
        self.load_error: Optional[str] = None
        self._model_lock = RLock()

    @property
    def model_available(self) -> bool:
        return self.model is not None and self.processor is not None

    def process_windows(
        self,
        video_id: str,
        windows: List[Dict[str, Any]],
        video_path: Path | str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Add notebook-schema events to windows, reusing any precomputed VLM events."""
        output_windows = deepcopy(windows)
        needs_inference = any(not window.get("events") for window in output_windows)
        model_available = self.model_available
        if needs_inference and not model_available:
            try:
                self._load_model()
                model_available = self.model_available
            except (ImportError, ModuleNotFoundError) as exc:
                self.load_error = str(exc)
                logger.warning(
                    "Qwen2.5-VL is unavailable; retaining grounded motion candidates "
                    "without marking them as VLM-verified: %s",
                    exc,
                )

        for index, window in enumerate(output_windows):
            start_sec, end_sec = map(float, window["window"])
            events = window.get("events") or []
            if events:
                window["events"] = self._normalize_existing_events(events, window, index)
                continue

            if model_available:
                images = self._extract_frames(
                    video_path,
                    start_sec,
                    end_sec,
                    int(config_loader.get("vlm.frames_per_window", 5)),
                )
                if images:
                    window["events"] = self._infer_window(window, images, index)
                    continue
                logger.warning("No frames extracted for %s window %s", video_id, window["window"])
            window["events"] = self._candidate_events(window, index)

        return output_windows, model_available

    def _load_model(self) -> None:
        with self._model_lock:
            if self.model_available:
                return
            self._load_model_unlocked()

    def _load_model_unlocked(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        load_options: Dict[str, Any] = {"device_map": "auto"}
        if self.execution_device.startswith("cuda"):
            load_options["torch_dtype"] = torch.float16
            if self._supports_4bit_quantization():
                from transformers import BitsAndBytesConfig

                load_options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                logger.warning(
                    "4-bit Qwen quantization requires bitsandbytes >= 0.46.1 on Linux; "
                    "loading in CUDA FP16 instead"
                )
        else:
            load_options["device_map"] = "cpu"
            load_options["torch_dtype"] = torch.float32

        logger.info("Loading %s for temporal event understanding", self.model_name)
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **load_options
            )
        except (ImportError, RuntimeError) as exc:
            message = str(exc).casefold()
            if "quantization_config" not in load_options or "bitsandbytes" not in message:
                raise
            logger.warning(
                "bitsandbytes could not initialize (%s); retrying Qwen in CUDA FP16",
                exc,
            )
            load_options.pop("quantization_config")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **load_options
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    @staticmethod
    def _supports_4bit_quantization() -> bool:
        """Use bitsandbytes only where its supported version is installed."""
        if not sys.platform.startswith("linux"):
            return False
        try:
            from packaging.version import Version

            return Version(version("bitsandbytes")) >= Version("0.46.1")
        except (ImportError, PackageNotFoundError, ValueError):
            return False

    def generate_grounded_answer(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> Optional[str]:
        """Answer conversational follow-ups from retrieved event records only."""
        if not evidence:
            return None

        evidence_ids = {
            str(item["event_id"])
            for item in evidence
            if item.get("event_id") is not None
        }
        if not evidence_ids:
            return None

        facts = [
            {
                "event_id": str(item["event_id"]),
                "video_id": item.get("video_id"),
                "event_type": item.get("event_type"),
                "start_sec": item.get("start_sec"),
                "end_sec": item.get("end_sec"),
                "description": item.get("description"),
                "confidence": item.get("confidence"),
                "anomaly_category": item.get("anomaly_category"),
                "anomaly_confidence": item.get("anomaly_confidence"),
            }
            for item in evidence[:10]
            if item.get("event_id") is not None
        ]
        recent_history = [
            turn for turn in history[-8:]
            if turn.get("role") in {"user", "assistant"}
            and isinstance(turn.get("content"), str)
        ]
        if os.environ.get("CHAT_PROVIDER", "qwen").strip().casefold() == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not api_key or api_key == "your_actual_api_key_here":
                logger.warning(
                    "CHAT_PROVIDER is groq but GROQ_API_KEY is not configured; "
                    "falling back to local Qwen"
                )
            else:
                try:
                    answer = self._generate_groq_answer(
                        question, facts, recent_history
                    )
                    if self._is_valid_groq_answer(answer):
                        return answer
                    logger.warning(
                        "Groq returned no final text answer; falling back to Qwen"
                    )
                except Exception:
                    logger.exception("Groq chat answer failed; falling back to local Qwen")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conversational assistant for surveillance video analysis. "
                    "Respond as a person would in a helpful back-and-forth conversation: "
                    "answer the current question directly in complete, natural sentences, "
                    "use prior turns to resolve follow-ups, and mention precise timestamps "
                    "when they help. Use only the VERIFIED EVENT RECORDS below for claims "
                    "about the footage. Treat "
                    "conversation history only as context for references such as 'it' or "
                    "'that person'; it is not evidence. Do not infer intent, identity, or "
                    "unstated actions. If the records do not answer the question, say so "
                    "conversationally. The interface displays the supporting event cards, "
                    "so do not output event IDs, labels, JSON, or a list of raw records."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history (context only):\n"
                    f"{json.dumps(recent_history, ensure_ascii=True)}\n\n"
                    f"Verified event records:\n{json.dumps(facts, ensure_ascii=True)}\n\n"
                    f"Current question: {question}"
                ),
            },
        ]

        return self._generate_qwen_answer(messages, evidence_ids)

    def _generate_qwen_answer(
        self,
        messages: List[Dict[str, str]],
        evidence_ids: set[str],
    ) -> Optional[str]:
        if not self.model_available:
            self._load_model()
        with self._model_lock:
            import torch

            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[prompt], return_tensors="pt")
            model_device = getattr(self.model, "device", None)
            if model_device is not None:
                inputs = inputs.to(model_device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=min(
                    256,
                    max(32, int(config_loader.get("chat.max_new_tokens", 192))),
                ),
                do_sample=False,
            )
            answer = self.processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

        citations = set(re.findall(r"\[([^\]]+)\]", answer))
        answer_without_citations = re.sub(r"\[[^\]]+\]", "", answer).strip()
        if not self._is_valid_chat_answer(answer_without_citations) or (
            citations and not citations.issubset(evidence_ids)
        ):
            logger.warning(
                "Rejected local Qwen chat answer because it was not a grounded conversational response"
            )
            return None
        return answer

    @staticmethod
    def _is_valid_chat_answer(answer: Optional[str]) -> bool:
        return bool(
            answer
            and len(answer.strip()) >= 24
            and re.search(r"[A-Za-z]{3}", answer)
        )

    @staticmethod
    def _is_valid_groq_answer(answer: Optional[str]) -> bool:
        return bool(answer and answer.strip())

    @staticmethod
    def _generate_groq_answer(
        question: str,
        facts: List[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> str:
        """Generate a fast cloud response using only the retrieved event records."""
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key or api_key == "your_actual_api_key_here":
            raise ValueError(
                "CHAT_PROVIDER is groq but GROQ_API_KEY is missing or still a placeholder"
            )
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        if not model:
            raise ValueError("GROQ_MODEL must not be empty")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful conversational assistant for surveillance video "
                    "analysis. Answer the current question directly in a clear, natural "
                    "explanation, usually 2–4 complete sentences with useful detail and "
                    "timestamps when relevant. Only verified event records are evidence "
                    "about the footage. Conversation history is context for follow-ups, "
                    "not evidence. Do not infer intent, identity, or unstated actions. If "
                    "the records do not answer the question, say so. Do not output event "
                    "IDs, JSON, or a list of raw records."
                ),
            },
            *history,
            {
                "role": "user",
                "content": (
                    f"Verified event records:\n{json.dumps(facts, ensure_ascii=True)}\n\n"
                    f"Current question: {question}"
                ),
            },
        ]
        client = Groq(api_key=api_key, max_retries=0, timeout=12.0)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=min(
                512, max(32, int(config_loader.get("chat.max_new_tokens", 384)))
            ),
            reasoning_effort="low",
            reasoning_format="hidden",
        )
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _extract_frames(
        video_path: Path | str,
        start_sec: float,
        end_sec: float,
        count: int,
    ):
        from PIL import Image

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for Qwen frames: {video_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not fps or not math.isfinite(fps) or fps <= 0:
                fps = 25.0
            if total_frames <= 0:
                return []
            last_sec = max(start_sec, min(end_sec, (total_frames - 1) / fps))
            first_sec = max(0.0, min(start_sec, last_sec))
            indices = np.linspace(
                round(first_sec * fps),
                round(last_sec * fps),
                num=max(1, count),
                dtype=int,
            )
            frames = []
            previous_index = -1
            for frame_index in indices:
                frame_index = int(min(frame_index, total_frames - 1))
                if frame_index == previous_index:
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if success and frame is not None:
                    max_dimension = int(
                        config_loader.get("vlm.max_frame_dimension", 768)
                    )
                    height, width = frame.shape[:2]
                    if max_dimension > 0 and max(height, width) > max_dimension:
                        scale = max_dimension / max(height, width)
                        frame = cv2.resize(
                            frame,
                            (round(width * scale), round(height * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                previous_index = frame_index
            return frames
        finally:
            capture.release()

    def _infer_window(
        self, window: Dict[str, Any], images, index: int
    ) -> List[Dict[str, Any]]:
        base_prompt = self._build_user_prompt(window)
        prompt = base_prompt
        last_error: Optional[Exception] = None
        retries = int(config_loader.get("vlm.max_retries", 2))
        for attempt in range(retries + 1):
            raw = self._call_model(images, prompt, sample=attempt > 0)
            try:
                return self._validate_response(raw, window, index)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                known_ids = sorted(int(track["track_id"]) for track in window["tracks"])
                prompt = (
                    f"{base_prompt}\n\nYour previous answer was rejected: {exc}. "
                    f"Valid track IDs: {known_ids}. Return corrected JSON only."
                )
                logger.warning(
                    "Rejected Qwen output for window %s (attempt %d/%d): %s",
                    window["window"], attempt + 1, retries + 1, exc,
                )
        logger.warning(
            "Dropping events for window %s after %d invalid Qwen responses: %s",
            window["window"], retries + 1, last_error,
        )
        return []

    def _call_model(
        self, images, prompt_text: str, sample: bool
    ) -> str:
        import torch

        user_content = [{"type": "image", "image": image} for image in images]
        user_content.append({"type": "text", "text": prompt_text})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)
        generation_options: Dict[str, Any] = {
            "max_new_tokens": int(config_loader.get("vlm.max_tokens", 1024)),
            "do_sample": sample,
        }
        if sample:
            generation_options["temperature"] = max(self.temperature, 0.2)
        with self._model_lock, torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_options)
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @staticmethod
    def _build_user_prompt(window: Dict[str, Any]) -> str:
        start_sec, end_sec = map(float, window["window"])
        lines = [
            f"Tracked objects in window {start_sec}-{end_sec}s "
            f"(window length {end_sec - start_sec:.2f}s):"
        ]
        candidates = []
        for track in window["tracks"]:
            track_id = int(track["track_id"])
            description = (
                f"- {track['class']} #{track_id}: "
                f"present {track.get('dwell_sec', 0)}s"
            )
            if track.get("zones"):
                description += f", zones: {' -> '.join(track['zones'])}"
            if track.get("held_by") is not None:
                description += f", held by #{track['held_by']}"
            if track.get("released_at") is not None:
                description += f", released at {track['released_at']}s"
            lines.append(description)
            candidates.extend(
                f"{event}({track_id})"
                for event in track.get("candidate_events", [])
            )
        lines.extend([
            "",
            f"Candidate events from motion rules: [{', '.join(candidates)}]",
            "Describe the actions in these frames. Schema: " + SCHEMA_HINT,
        ])
        return "\n".join(lines)

    @classmethod
    def _validate_response(
        cls, raw_text: str, window: Dict[str, Any], index: int
    ) -> List[Dict[str, Any]]:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        if not cleaned.startswith("{"):
            first_object = cleaned.find("{")
            last_object = cleaned.rfind("}")
            if first_object < 0 or last_object < first_object:
                raise ValueError("Model response does not contain a JSON object")
            cleaned = cleaned[first_object:last_object + 1]
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            parsed = {"events": parsed}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
            raise ValueError("Response must contain a top-level events array")

        start_sec, end_sec = map(float, window["window"])
        duration = end_sec - start_sec
        known_ids = {int(track["track_id"]) for track in window["tracks"]}
        events = []
        for event_index, event in enumerate(parsed["events"]):
            if not isinstance(event, dict):
                raise ValueError(f"Event {event_index} is not an object")
            event_type = event.get("event_type", "scene_description")
            description = event.get("description")
            if not isinstance(event_type, str) or not event_type.strip():
                raise ValueError(f"Event {event_index} has no event_type")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Event {event_index} has no description")
            subject = cls._track_id(event.get("subject_track"), known_ids, "subject_track")
            object_track = cls._track_id(event.get("object_track"), known_ids, "object_track")
            relative_start = cls._finite_float(event.get("start", 0.0), "start")
            relative_end = cls._finite_float(event.get("end", duration), "end")
            confidence = cls._finite_float(event.get("confidence", 0.8), "confidence")
            if relative_start < 0 or relative_end < relative_start or relative_end > duration + 1:
                raise ValueError(f"Event {event_index} times are outside this temporal window")
            if not 0 <= confidence <= 1:
                raise ValueError(f"Event {event_index} confidence must be between 0 and 1")
            events.append({
                "event_type": event_type.strip(),
                "subject_track": subject,
                "object_track": object_track,
                "start": relative_start,
                "end": relative_end,
                "abs_start": start_sec + relative_start,
                "abs_end": start_sec + relative_end,
                "zone": event.get("zone"),
                "confidence": confidence,
                "description": description.strip(),
                "vlm_verified": True,
            })
        return events

    @classmethod
    def _normalize_existing_events(
        cls, events: List[Dict[str, Any]], window: Dict[str, Any], index: int
    ) -> List[Dict[str, Any]]:
        start_sec, end_sec = map(float, window["window"])
        duration = end_sec - start_sec
        known_ids = {int(track["track_id"]) for track in window["tracks"]}
        normalized = []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"Window {index} event {event_index} must be an object")
            event_type = event.get("event_type")
            description = event.get("description")
            if not isinstance(event_type, str) or not event_type.strip():
                raise ValueError(f"Window {index} event {event_index} needs event_type")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Window {index} event {event_index} needs description")
            subject = cls._track_id(event.get("subject_track"), known_ids, "subject_track")
            object_track = cls._track_id(event.get("object_track"), known_ids, "object_track")

            if event.get("abs_start") is not None and event.get("abs_end") is not None:
                abs_start = cls._finite_float(event["abs_start"], "abs_start")
                abs_end = cls._finite_float(event["abs_end"], "abs_end")
                relative_start, relative_end = abs_start - start_sec, abs_end - start_sec
            else:
                relative_start = cls._finite_float(event.get("start", 0.0), "start")
                relative_end = cls._finite_float(event.get("end", duration), "end")
                abs_start, abs_end = start_sec + relative_start, start_sec + relative_end
            if relative_start < 0 or relative_end < relative_start or relative_end > duration + 1:
                raise ValueError(f"Window {index} event {event_index} is outside its time range")
            confidence = cls._finite_float(event.get("confidence", 0.8), "confidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"Window {index} event {event_index} confidence must be between 0 and 1")

            normalized.append({
                **event,
                "event_type": event_type.strip(),
                "subject_track": subject,
                "object_track": object_track,
                "start": relative_start,
                "end": relative_end,
                "abs_start": abs_start,
                "abs_end": abs_end,
                "confidence": confidence,
                "description": description.strip(),
                "vlm_verified": bool(event.get("vlm_verified", False)),
            })
        return normalized

    @classmethod
    def _candidate_events(
        cls, window: Dict[str, Any], index: int
    ) -> List[Dict[str, Any]]:
        start_sec, end_sec = map(float, window["window"])
        candidates = []
        for track in window["tracks"]:
            track_id = int(track["track_id"])
            for candidate in track.get("candidate_events", []):
                if not isinstance(candidate, str) or not candidate:
                    continue
                candidates.append({
                    "event_type": candidate,
                    "subject_track": track_id,
                    "object_track": None,
                    "start": 0.0,
                    "end": end_sec - start_sec,
                    "abs_start": start_sec,
                    "abs_end": end_sec,
                    "zone": (track.get("zones") or [None])[0],
                    "confidence": 0.5,
                    "description": (
                        f"{track['class']} #{track_id} is associated with a "
                        f"{candidate.replace('_', ' ')} motion-rule candidate."
                    ),
                    "vlm_verified": False,
                    "source": "motion_rule_fallback",
                })
        return candidates

    @staticmethod
    def _finite_float(value: Any, field: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field} must be a finite number")
        return number

    @staticmethod
    def _track_id(value: Any, known_ids: set[int], field: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer track ID or null")
        try:
            track_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer track ID or null") from exc
        if track_id not in known_ids:
            raise ValueError(f"{field} {track_id} is not present in this temporal window")
        return track_id
