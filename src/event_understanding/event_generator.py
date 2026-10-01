"""Stage 2 Temporal Windowing and Candidate Event Generator."""

from typing import List, Dict, Any, Tuple
import math
from src.tracking.schemas import Track, TrackingResult
from src.event_understanding.schemas import (
    CandidateEvent,
    TemporalWindow,
)
from src.utils.logger import logger
from src.config.loader import config_loader


class TemporalWindower:
    """Groups tracks into overlapping temporal windows and extracts rule-based candidate events."""

    def __init__(
        self,
        window_duration_sec: float = 10.0,
        window_stride_sec: float = 5.0,
        loiter_dwell_sec: float = 4.0,
        loiter_displacement_ratio: float = 0.35,
        proximity_scale: float = 1.0,
    ):
        self.window_duration_sec = float(window_duration_sec)
        self.window_stride_sec = float(window_stride_sec)
        self.loiter_dwell_sec = float(loiter_dwell_sec)
        self.loiter_displacement_ratio = float(loiter_displacement_ratio)
        self.proximity_scale = float(proximity_scale)

    def generate_windows(
        self, tracking_result: TrackingResult, video_duration_sec: float
    ) -> List[TemporalWindow]:
        """Generate sliding windows and candidate events for all tracks in a video."""
        tracks = tracking_result.tracks
        if not tracks and video_duration_sec <= 0:
            return []

        # Compute median person height for normalized spatial distances
        person_heights = [
            t.median_height for t in tracks if t.class_name.lower() == "person"
        ]
        if person_heights:
            person_heights.sort()
            median_person_h = float(person_heights[len(person_heights) // 2])
        else:
            median_person_h = 100.0  # reasonable fallback pixel height

        proximity_px = median_person_h * self.proximity_scale

        # Generate window boundaries [w_start, w_end]
        max_time = max(
            video_duration_sec,
            max((t.end_sec for t in tracks), default=0.0),
        )
        windows: List[TemporalWindow] = []
        w_start = 0.0
        w_idx = 1

        while w_start < max_time or w_idx == 1:
            w_end = min(w_start + self.window_duration_sec, max_time)
            if w_end <= w_start:
                break

            # Find tracks active in this window
            active_tracks = [
                t
                for t in tracks
                if not (t.end_sec < w_start or t.start_sec > w_end)
            ]
            active_tids = [int(t.track_id) for t in active_tracks]

            candidate_events: List[CandidateEvent] = []

            # Process tracks inside this window
            for track in active_tracks:
                tid = int(track.track_id)
                cname = track.class_name.lower()

                # Rule 1: Enter event (fire only if track globally starts within this window)
                if w_start <= track.start_sec < w_end:
                    candidate_events.append(
                        CandidateEvent(
                            event_type="enter",
                            start_sec=float(track.start_sec),
                            end_sec=float(min(track.start_sec + 1.5, track.end_sec)),
                            entity_ids=[tid],
                            confidence=0.90,
                            details={"class_name": track.class_name},
                        )
                    )

                # Rule 2: Exit event (fire only if track globally ends within this window)
                if w_start < track.end_sec <= w_end and track.end_sec < max_time - 0.5:
                    candidate_events.append(
                        CandidateEvent(
                            event_type="exit",
                            start_sec=float(max(track.end_sec - 1.5, track.start_sec)),
                            end_sec=float(track.end_sec),
                            entity_ids=[tid],
                            confidence=0.88,
                            details={"class_name": track.class_name},
                        )
                    )

                # Observations inside this window
                win_obs = [
                    obs
                    for obs in track.observations
                    if w_start <= obs.timestamp_sec <= w_end
                ]
                if len(win_obs) >= 3:
                    sub_duration = win_obs[-1].timestamp_sec - win_obs[0].timestamp_sec
                    sub_dx = win_obs[-1].bbox.center_x - win_obs[0].bbox.center_x
                    sub_dy = win_obs[-1].bbox.center_y - win_obs[0].bbox.center_y
                    sub_displacement = math.sqrt(sub_dx**2 + sub_dy**2)

                    # Rule 3: Loitering (Person remains for > loiter_dwell_sec with low net displacement)
                    if (
                        cname == "person"
                        and sub_duration >= self.loiter_dwell_sec
                        and sub_displacement < (median_person_h * 1.5)
                    ):
                        candidate_events.append(
                            CandidateEvent(
                                event_type="loiter",
                                start_sec=float(win_obs[0].timestamp_sec),
                                end_sec=float(win_obs[-1].timestamp_sec),
                                entity_ids=[tid],
                                confidence=0.85,
                                details={
                                    "dwell_sec": round(float(sub_duration), 2),
                                    "displacement_px": round(float(sub_displacement), 2),
                                },
                            )
                        )

                    # Rule 4: Stationary (low displacement for any entity)
                    if sub_duration >= 3.0 and sub_displacement < (median_person_h * 0.3):
                        candidate_events.append(
                            CandidateEvent(
                                event_type="stationary",
                                start_sec=float(win_obs[0].timestamp_sec),
                                end_sec=float(win_obs[-1].timestamp_sec),
                                entity_ids=[tid],
                                confidence=0.80,
                                details={"displacement_px": round(float(sub_displacement), 2)},
                            )
                        )

            # Rule 5 & 6: Object Interactions (Carry, Place, Leave) between Person and Objects
            persons = [t for t in active_tracks if t.class_name.lower() == "person"]
            objects = [
                t
                for t in active_tracks
                if t.class_name.lower() in {"backpack", "handbag", "suitcase", "bicycle"}
            ]

            for person in persons:
                for obj in objects:
                    # Check spatial proximity over overlapping timestamps
                    p_obs_map = {obs.frame_idx: obs for obs in person.observations}
                    common_frames = [
                        obs for obs in obj.observations if obs.frame_idx in p_obs_map
                    ]
                    if not common_frames:
                        continue

                    dists = []
                    for o_obs in common_frames:
                        p_obs = p_obs_map[o_obs.frame_idx]
                        dist = math.sqrt(
                            (o_obs.bbox.center_x - p_obs.bbox.center_x) ** 2
                            + (o_obs.bbox.center_y - p_obs.bbox.center_y) ** 2
                        )
                        dists.append((dist, o_obs.timestamp_sec))

                    if not dists:
                        continue

                    min_dist = min(d[0] for d in dists)
                    if min_dist <= proximity_px:
                        start_inter = float(common_frames[0].timestamp_sec)
                        end_inter = float(common_frames[-1].timestamp_sec)

                        # If both are moving together -> Carry
                        if (
                            person.displacement > median_person_h
                            and obj.displacement > median_person_h
                        ):
                            candidate_events.append(
                                CandidateEvent(
                                    event_type="carry",
                                    start_sec=start_inter,
                                    end_sec=end_inter,
                                    entity_ids=[int(person.track_id), int(obj.track_id)],
                                    confidence=0.82,
                                    details={"min_dist_px": round(float(min_dist), 2)},
                                )
                            )
                        # If person departs while object stays -> Place / Leave
                        elif person.displacement > median_person_h and obj.displacement < (
                            median_person_h * 0.4
                        ):
                            candidate_events.append(
                                CandidateEvent(
                                    event_type="place",
                                    start_sec=start_inter,
                                    end_sec=min(start_inter + 3.0, end_inter),
                                    entity_ids=[int(person.track_id), int(obj.track_id)],
                                    confidence=0.84,
                                    details={"min_dist_px": round(float(min_dist), 2)},
                                )
                            )

            windows.append(
                TemporalWindow(
                    window_id=w_idx,
                    start_sec=round(float(w_start), 2),
                    end_sec=round(float(w_end), 2),
                    track_ids=active_tids,
                    candidate_events=candidate_events,
                )
            )

            w_start += self.window_stride_sec
            w_idx += 1

        # Global Dedup pass across overlapping windows for enter/exit
        windows = self._deduplicate_boundary_events(windows)

        logger.info(
            f"Generated {len(windows)} temporal windows with {sum(len(w.candidate_events) for w in windows)} candidate events"
        )
        return windows

    def _deduplicate_boundary_events(
        self, windows: List[TemporalWindow]
    ) -> List[TemporalWindow]:
        """Ensure enter and exit events fire at most once globally per track entity."""
        seen_enters = set()
        seen_exits = set()

        for win in windows:
            filtered_events = []
            for ev in win.candidate_events:
                if ev.event_type == "enter":
                    tid = ev.entity_ids[0] if ev.entity_ids else None
                    if tid in seen_enters:
                        continue
                    seen_enters.add(tid)
                elif ev.event_type == "exit":
                    tid = ev.entity_ids[0] if ev.entity_ids else None
                    if tid in seen_exits:
                        continue
                    seen_exits.add(tid)
                filtered_events.append(ev)
            win.candidate_events = filtered_events

        return windows
