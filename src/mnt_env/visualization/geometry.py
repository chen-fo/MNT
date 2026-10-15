"""Render geometry from detached debug snapshots; no simulator mutations."""
import numpy as np

from mnt_env.env import DELTAS


def shot_segments(snapshot):
    names = snapshot["agent_order"]
    positions = snapshot["positions"]
    size = snapshot["config"]["size"]
    segments = []
    for event in snapshot["events"]:
        if event["kind"] != "shot":
            continue
        start = positions[names.index(event["shooter"])].copy()
        if event["blockers"]:
            end = positions[names.index(event["blockers"][0])].copy()
        else:
            end = start.copy()
            for distance in range(1, snapshot["config"]["shoot_range"] + 1):
                point = start + DELTAS[event["bearing"]] * distance
                if not np.all((point >= 0) & (point < size)):
                    break
                end = point
        segments.append((start, end, event["victim"] is not None))
    return segments
