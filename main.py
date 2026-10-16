#!/usr/bin/env python3
"""Play until the session end, counting only during the configured interval."""
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from counting import LaneCounter
from utils import draw_analytics, dumps_json, person_role, iou, intersection, area

# Session settings: edit these two values (seconds).
SESSION_START_SECONDS = 6.0
SESSION_END_SECONDS = 60.0

# Playback settings. Space pauses; Q or Esc stops and saves.
SHOW_WINDOW = True
SAVE_VIDEO = True
DEVICE = 'auto'  # '0' requires CUDA; 'cpu' forces CPU.
IMAGE_SIZE = 960
WINDOW_NAME = 'Checkout'

# Input files are relative to this script, independent of the working directory.
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / '66ep_46ep.pt'
PERSON_MODEL_PATH = BASE / 'yolo26n.pt'
VIDEO_PATH = BASE / 'videoplayback.mp4'
ROI_PATH = BASE / 'roi_config.yml'

# Base detector crops; expand them to cover the configured bagging area.
CROPS = {
    'center': (760, 140, 1120, 820),
    'right': (1320, 140, 1840, 800),
}
ROI_NAMES = {
    'conveyor': 'CONVEYOR',
    'scanner': 'SCANNER',
    'bagging': 'BAGGING_AREA',
}


def load_rois(path=ROI_PATH):
    """A missing CONVEYOR key disables its lane; enabled lanes need all ROIs."""
    config = yaml.safe_load(Path(path).read_text()) or {}
    config = {name.upper().replace(' ', '_'): points for name, points in config.items()}
    lanes = {}

    for lane, suffix in [('center', ''), ('right', '_2')]:
        if 'CONVEYOR' + suffix not in config:
            continue

        zones = {}
        for zone, name in ROI_NAMES.items():
            key = name + suffix
            if key not in config:
                raise ValueError(
                    f'{lane}: missing {key}. Provide all three ROIs '
                    'or comment out the lane\'s CONVEYOR key.'
                )
            polygon = np.asarray(config[key], np.float32)
            if (
                polygon.ndim != 2
                or polygon.shape[1] != 2
                or len(polygon) < 3
                or cv2.contourArea(polygon) <= 0
            ):
                raise ValueError(f'Invalid ROI: {key}')
            zones[zone] = polygon
        lanes[lane] = zones

    if not lanes:
        raise ValueError('No enabled conveyor found in roi_config.yml')
    return lanes


def in_session(second):
    """The start is inclusive and the end is exclusive."""
    return SESSION_START_SECONDS <= second < SESSION_END_SECONDS


def load_person_rois(enabled_lanes, path=ROI_PATH):
    """Optional role polygons, kept separate from item-counting polygons."""
    config = yaml.safe_load(Path(path).read_text()) or {}
    config = {name.upper().replace(' ', '_'): points for name, points in config.items()}
    regions = []
    for lane, suffix in [('center', ''), ('right', '_2')]:
        if lane not in enabled_lanes:
            continue
        for role, names in [
            ('customer', ['CUSTOMER_ROI', 'CUSTOMER_SIDE']),
            ('cashier', ['CASHIER_ROI']),
        ]:
            key = next((name + suffix for name in names if name + suffix in config), None)
            if key is None:
                continue
            polygon = np.asarray(config[key], np.float32)
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                    or not np.isfinite(polygon).all() or cv2.contourArea(polygon) <= 0):
                raise ValueError(f'Invalid person ROI: {key}')
            regions.append({'lane': lane, 'role': role, 'polygon': polygon})
    return regions


def detect_people(model, device, frame, regions):
    """Use the whole frame so the person's feet are not cut off by item crops."""
    if model is None:
        return []
    result = model.predict(
        frame, classes=[0], conf=0.35, imgsz=IMAGE_SIZE,
        device=device, verbose=False, max_det=100,
    )[0]
    people = []
    for x1, y1, x2, y2, confidence, _ in result.boxes.data.cpu().tolist():
        box = [x1, y1, x2, y2]
        region = person_role(box, regions)
        if region is not None:
            people.append({'box': box, 'role': region['role'], 'lane': region['lane']})
    return people


def load_model():
    """Configure writable runtime folders and select the inference device."""
    for key, folder in [
        ('YOLO_CONFIG_DIR', '.runtime/ultralytics'),
        ('MPLCONFIGDIR', '.runtime/matplotlib'),
    ]:
        (BASE / folder).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(BASE / folder))

    import torch
    from ultralytics import YOLO

    device = DEVICE
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(min(8, os.cpu_count() or 1))

    model = YOLO(str(MODEL_PATH))
    if model.names != {0: 'retail_item', 1: 'carry_bag'}:
        raise ValueError('Unexpected model classes')
    return model, device


def detection_crop(lane, rois, frame_shape):
    """Include the bagging ROI plus room for a held product at its edge."""
    x1, y1, x2, y2 = CROPS[lane]
    points = np.asarray(rois['bagging'])
    lower = np.floor((points.min(axis=0) - 40) / 10) * 10
    upper = np.ceil((points.max(axis=0) + 40) / 10) * 10
    height, width = frame_shape[:2]
    return (max(0, min(x1, int(lower[0]))),
            max(0, min(y1, int(lower[1]))),
            min(width, max(x2, int(upper[0]))),
            min(height, max(y2, int(upper[1]))))


def merge_detections(primary, extra):
    """Keep the training-crop detections; add nonduplicate packing-area boxes."""
    merged = list(primary)
    for box in sorted(extra, key=lambda row: -row[4]):
        if box[4] < .35:
            continue
        duplicate = any(
            int(box[5]) == int(other[5]) and
            (iou(box, other) >= .25 or
             intersection(box, other) / min(area(box), area(other)) >= .50)
            for other in merged
        )
        if not duplicate:
            merged.append(box)
    return np.asarray(merged, np.float32).reshape(-1, 6)


def predict_crop(model, device, frame, crop):
    x1, y1, x2, y2 = crop
    result = model.predict(frame[y1:y2, x1:x2], imgsz=IMAGE_SIZE,
                           conf=.10, device=device, verbose=False, max_det=100)[0]
    detections = result.boxes.data.cpu().numpy().copy()
    if len(detections):
        detections[:, [0, 2]] += x1
        detections[:, [1, 3]] += y1
    return detections


def analyze_frame(model, device, counters, frame, second):
    """Keep familiar detector scale and also inspect packing outside its crop."""
    for lane, counter in counters.items():
        base_crop = CROPS[lane]
        detections = predict_crop(model, device, frame, base_crop)
        expanded = detection_crop(lane, counter.rois, frame.shape)
        if expanded != base_crop:
            extra = predict_crop(model, device, frame, expanded)
            detections = merge_detections(detections, extra)
        counter.track(detections, frame, second)


def draw(frame, counters, rois, second, last_analysis, person_rois=(), people=()):
    """Rendering never updates the counters."""
    draw_analytics(
        frame, counters, rois, second, last_analysis,
        SESSION_START_SECONDS, SESSION_END_SECONDS,
        person_rois=person_rois, people=people,
    )


def show_frame(frame, fps, frame_started):
    """Display a frame, handle pause, and return whether playback should continue."""
    cv2.imshow(WINDOW_NAME, frame)
    remaining = 1 / fps - (time.perf_counter() - frame_started)
    key = cv2.waitKey(max(1, round(1000 * remaining))) & 255

    if key == ord(' '):
        while True:
            key = cv2.waitKey(30) & 255
            if key in (ord(' '), ord('q'), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return False

    return (
        key not in (ord('q'), 27)
        and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    )


def save_results(output, counters, last_analysis, analyzed_frames, played_until):
    """Save estimates and their evidence, including interrupted sessions."""
    result = {
        'session_start_seconds': SESSION_START_SECONDS,
        'session_end_seconds': SESSION_END_SECONDS,
        'last_analyzed_seconds': last_analysis if analyzed_frames else None,
        'analyzed_frames': analyzed_frames,
        'session_completed': played_until >= SESSION_END_SECONDS,
        'video_played_until_seconds': played_until,
        'lanes': {lane: counter.summary(last_analysis) for lane, counter in counters.items()},
       
    }
    artifacts = {
        'results.json': result,
        'tracks.json': {lane: counter.export_tracks() for lane, counter in counters.items()},
        'events.json': {lane: counter.events for lane, counter in counters.items()},
    }
    # Serialize everything before writing so a type error cannot leave a partial export.
    serialized = {filename: dumps_json(content) for filename, content in artifacts.items()}
    for filename, content in serialized.items():
        (output / filename).write_text(content)
    (BASE / 'outputs/latest_run.txt').write_text(str(output) + '\n')
    print(dumps_json(result), flush=True)


def main():
    rois = load_rois()
    person_rois = load_person_rois(rois)
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if not capture.isOpened() or fps <= 0:
            raise ValueError(f'Cannot open video: {VIDEO_PATH}')
        if not 0 <= SESSION_START_SECONDS < SESSION_END_SECONDS <= total_frames / fps:
            raise ValueError(
                f'Session must satisfy 0 <= start < end <= {total_frames / fps:.2f} seconds'
            )
        if frame_size != (1920, 1080):
            raise ValueError('These ROI/crop coordinates require the original 1920x1080 video.')
        has_display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
        if SHOW_WINDOW and os.name != 'nt' and not has_display:
            raise RuntimeError('No desktop display. Use a desktop terminal or set SHOW_WINDOW=False.')

        model, device = load_model()
        person_model = None
        if person_rois:
            from ultralytics import YOLO
            if not PERSON_MODEL_PATH.is_file():
                raise FileNotFoundError(PERSON_MODEL_PATH)
            person_model = YOLO(str(PERSON_MODEL_PATH))
            if person_model.names.get(0) != 'person':
                raise ValueError('Person model must have class 0 named person')
        counters = {
            lane: LaneCounter(zones, fps, SESSION_START_SECONDS)
            for lane, zones in rois.items()
        }
        output = BASE / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True)
        if SAVE_VIDEO:
            writer = cv2.VideoWriter(
                str(output / 'annotated.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                frame_size,
            )
            if not writer.isOpened():
                raise RuntimeError('Could not create video output')
        if SHOW_WINDOW:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)

        print(
            f'Enabled lanes: {list(rois)}; '
            f'session: {SESSION_START_SECONDS}-{SESSION_END_SECONDS}s; device: {device}',
            flush=True,
        )
        print(f'Output: {output}', flush=True)
        frame_index = 0
        analyzed_frames = 0
        last_analysis = SESSION_START_SECONDS

        while frame_index < total_frames and frame_index / fps < SESSION_END_SECONDS:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Video decode failed at frame {frame_index}')
            second = frame_index / fps
            people = []

            if in_session(second):
                analyze_frame(model, device, counters, frame, second)
                people = detect_people(person_model, device, frame, person_rois)
                last_analysis = second
                analyzed_frames += 1

            draw(frame, counters, rois, second, last_analysis, person_rois, people)
            if writer is not None and in_session(second):
                writer.write(frame)
            frame_index += 1
            if frame_index % round(fps * 10) == 0:
                print(f'Video {frame_index / fps:.0f}/{total_frames / fps:.0f}s', flush=True)
            if SHOW_WINDOW and not show_frame(frame, fps, frame_started):
                break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    # Finalize the video and close the window before exporting the counts.
    save_results(output, counters, last_analysis, analyzed_frames, frame_index / fps)


if __name__ == '__main__':
    main()
