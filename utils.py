"""Shared JSON, ROI geometry, appearance and video drawing helpers."""
import json
import cv2
import numpy as np

def json_default(value):
    """Convert NumPy values without silently stringifying unsupported objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_json(value):
    """One serialization path for results, tracks, events and console output."""
    return json.dumps(value, indent=2, default=json_default)


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def area(b):
    return max(1, (b[2]-b[0])*(b[3]-b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / (area(a)+area(b)-overlap)


def inside(box, polygon):
    """Accept the box center OR bottom-center, including polygon boundaries."""
    middle_x = float((box[0] + box[2]) / 2)
    center = (middle_x, float((box[1] + box[3]) / 2))
    bottom_center = (middle_x, float(box[3]))
    return (
        cv2.pointPolygonTest(polygon, center, False) >= 0
        or cv2.pointPolygonTest(polygon, bottom_center, False) >= 0
    )


def touches_roi(box, polygon):
    if inside(box, polygon):
        return True
    x1,y1,x2,y2 = box
    rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
    overlap = cv2.intersectConvexConvex(cv2.convexHull(polygon),rect)[0]
    return overlap/area(box) >= .20


def overlaps_bag(item, bag):
    # Item coverage works better than IoU alone when the bag is much bigger.
    return intersection(item, bag)/area(item) >= .25 or iou(item, bag) >= .08


def appearance(frame, box):
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = map(int, box)
    patch = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    hist = np.zeros(96, np.float32)
    if patch.size:
        hist = cv2.calcHist([cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)], [0,1], None,
                           [12,8], [0,180,0,256]).flatten()
        hist /= max(float(np.linalg.norm(hist)), 1e-6)
    return hist


# BGR colors, chosen for strong contrast on video.
BACKGROUND = (30, 24, 20)
CARD = (49, 41, 34)
BORDER = (83, 70, 58)
TEXT = (247, 244, 240)
MUTED = (190, 177, 162)
TEAL = (195, 215, 65)
AMBER = (75, 190, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(image, label, position, scale=0.6, color=TEXT, thickness=1):
    cv2.putText(image, str(label), position, FONT, scale, color, thickness, cv2.LINE_AA)


def timestamp(seconds):
    minutes, remaining = divmod(max(0, seconds), 60)
    return f"{int(minutes):02d}:{remaining:04.1f}"


def value_label(value, waiting=False):
    return "--" if waiting or value is None else str(value)


def metric_card(panel, x, y, width, label, value, accent=TEXT):
    cv2.rectangle(panel, (x, y), (x + width, y + 82), CARD, -1)
    cv2.rectangle(panel, (x, y), (x + 3, y + 82), accent, -1)
    text(panel, label, (x + 14, y + 24), 0.55, MUTED)
    text(panel, value, (x + 14, y + 65), 1.15, accent, 2)


def draw_panel(frame, summaries, second, start, end):
    """Keep the panel left of both checkout crops in the supplied 1080p video."""
    waiting = second < start
    active = start <= second < end
    status = "WAITING" if waiting else ("COUNTING" if active else "COMPLETE")
    status_color = MUTED if waiting else (TEAL if active else AMBER)
    width = 700
    header_height, lane_height, footer_height = 110, 244, 56
    height = header_height + lane_height * len(summaries) + footer_height
    panel = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), BORDER, 1)

    text(panel, "CHECKOUT ANALYTICS", (22, 35), 0.85, TEXT, 2)
    cv2.circle(panel, (527, 28), 5, status_color, -1, cv2.LINE_AA)
    text(panel, status, (543, 35), 0.63, status_color, 2)
    text(panel, f"VIDEO  {timestamp(second)}", (22, 68), 0.62, TEXT)
    text(panel, f"SESSION  {timestamp(start)} - {timestamp(end)}", (270, 68), 0.57, MUTED)
    progress = min(1.0, max(0.0, (second - start) / max(end - start, 0.001)))
    cv2.rectangle(panel, (22, 87), (width - 22, 92), CARD, -1)
    if progress:
        cv2.rectangle(panel, (22, 87), (22 + int((width - 44) * progress), 92), status_color, -1)

    for index, (lane, summary) in enumerate(summaries.items()):
        top = header_height + index * lane_height
        text(panel, f"{lane.upper()} CHECKOUT", (22, top + 24), 0.65, TEAL, 2)
        initial = value_label(summary["conveyor_at_start"], waiting)
        text(panel, f"On conveyor at start: {initial}", (355, top + 24), 0.55, MUTED)
        metrics = [
            ("CONVEYOR", "conveyor_unique", TEXT),
            ("SCANNER", "scanner_unique", TEXT),
            ("BAGGING AREA", "bagging_area_unique", TEXT),
            ("BAGGED (EST.)", "bagged_items_estimate", AMBER),
            ("BAGS USED", "bags_used_unique", TEAL),
            ("BAGS PRESENT" if active or waiting else "BAGS AT END", "bags_present_at_end", TEAL),
        ]
        for metric_index, (label, key, accent) in enumerate(metrics):
            row, column = divmod(metric_index, 3)
            metric_card(panel, 22 + column * 222, top + 40 + row * 94, 212,
                        label, value_label(summary[key], waiting), accent)
        cv2.line(panel, (22, top + 232), (width - 22, top + 232), BORDER, 1)

    footer_y = height - footer_height
    text(panel, "Stage counts are unique visits. Packing is estimated; -- = unavailable.",
         (22, footer_y + 16), 0.48, MUTED)
    text(panel, "SPACE  pause / resume     Q / ESC  save and exit", (22, footer_y + 41), 0.52, TEXT)
    # Scale down only for previews smaller than the original video.
    scale = min(1.0, (frame.shape[1] - 40) / width, (frame.shape[0] - 40) / height)
    if scale < 1:
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    panel_height, panel_width = panel.shape[:2]
    frame[20:20 + panel_height, 20:20 + panel_width] = panel


# OpenCV BGR colors. Adjust alpha here without changing counting behavior.
ROI_STYLES = {
    'conveyor': {'color': (230, 173, 82), 'fill': (173, 100, 35), 'alpha': 0.14},
    'scanner': {'color': (117, 222, 149), 'fill': (30, 65, 34), 'alpha': 0.48},
    'bagging': {'color': (218, 151, 211), 'fill': (130, 55, 117), 'alpha': 0.14},
    'customer': {'color': (110, 216, 244), 'fill': (55, 139, 160), 'alpha': 0.07},
    'cashier': {'color': (118, 161, 248), 'fill': (45, 78, 157), 'alpha': 0.07},
}


def tint_polygon(frame, polygon, color, alpha):
    """Fill a polygon on a small overlay, then blend without hiding the footage."""
    points = np.asarray(polygon, np.int32)
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x), max(0, y)
    right, bottom = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if right <= left or bottom <= top:
        return
    patch = frame[top:bottom, left:right]
    overlay = patch.copy()
    local_points = points - np.array([left, top])
    cv2.fillPoly(overlay, [local_points], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, patch, 1 - alpha, 0, dst=patch)


def label_chip(frame, label, x, y, color, scale=0.55):
    """Consistent, anti-aliased white text on a filled dark label background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (width, height), baseline = cv2.getTextSize(label, font, scale, 1)
    chip_width, chip_height = width + 22, height + baseline + 14
    x = max(0, min(int(x), frame.shape[1] - chip_width))
    y = max(chip_height, min(int(y), frame.shape[0]))
    cv2.rectangle(frame, (x, y - chip_height), (x + chip_width, y), (34, 29, 25), -1)
    cv2.rectangle(frame, (x, y - chip_height), (x + 4, y), color, -1)
    cv2.putText(frame, label, (x + 12, y - baseline - 7), font, scale,
                (248, 246, 243), 1, cv2.LINE_AA)


def roi_label(frame, label, polygon, color):
    """Right-align the label just above the polygon's top-right corner."""
    points = np.asarray(polygon)
    top_right = np.array([points[:, 0].max(), points[:, 1].min()])
    corner = points[np.argmin(np.sum((points - top_right) ** 2, axis=1))]
    width = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)[0][0] + 22
    label_chip(frame, label, int(corner[0]) - width, int(corner[1]) - 5, color, 0.48)


def draw_detection(frame, box, label, color, person=False):
    x1, y1, x2, y2 = map(round, box)
    polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    tint_polygon(frame, polygon, color, 0.035 if person else 0.09)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    # Short, heavier corners make boxes clear without a thick rectangle over the object.
    length = max(3, min(14, (x2 - x1) // 4, (y2 - y1) // 4))
    for x, y, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                          (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (x, y), (x + dx * length, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * length), color, 2, cv2.LINE_AA)
    label_chip(frame, label, x1, y1 - 3, color, 0.63 if person else 0.52)


def draw_roi_fills(frame, rois, person_rois):
    # Draw all fills before any text or boxes. Scanner goes last for stronger emphasis.
    regions = [(region['role'], region['polygon']) for region in person_rois]
    regions += [(name, polygon) for zones in rois.values() for name, polygon in zones.items()]
    for name, polygon in sorted(regions, key=lambda entry: entry[0] == 'scanner'):
        style = ROI_STYLES[name]
        tint_polygon(frame, polygon, style['fill'], style['alpha'])


def draw_scene(frame, counters, rois, second, active):
    for lane, zones in rois.items():
        for name, polygon in zones.items():
            roi_color = ROI_STYLES[name]['color']
            cv2.polylines(frame, [polygon.astype(np.int32)], True, roi_color, 2 if name == 'scanner' else 1, cv2.LINE_AA)
            label = "BAGGING AREA" if name == "bagging" else name.upper()
            roi_label(frame, label, polygon, roi_color)
        if not active:
            continue
        for tracked_object in counters[lane].objects.values():
            age = second - tracked_object["last"]
            if not tracked_object["confirmed"] or age > 0.4 or not tracked_object["zones"]:
                continue
            x1, y1, x2, y2 = map(round, tracked_object["box"])
            is_item = tracked_object["cls"] == 0
            color = TEAL if is_item else AMBER
            label = f"{lane[0].upper()}-{tracked_object['id']} {'ITEM' if is_item else 'BAG'}"
            if age > 0.1:
                label += " (held)"
            draw_detection(frame, tracked_object['box'], label, color)


def person_role(box, regions):
    """Person filtering uses ONLY bottom-center, never center or box overlap.

    For overlapping regions, choose the one with the largest distance from its edge.
    """
    point = (float((box[0] + box[2]) / 2), float(box[3]))
    matches = [
        (cv2.pointPolygonTest(region['polygon'], point, True), region)
        for region in regions
    ]
    matches = [(distance, region) for distance, region in matches if distance >= 0]
    return max(matches, key=lambda match: match[0])[1] if matches else None


def draw_people(frame, regions, people):
    colors = {role: ROI_STYLES[role]['color'] for role in ('customer', 'cashier')}
    for region in regions:
        polygon = region['polygon'].astype(np.int32)
        color = colors[region['role']]
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        roi_label(frame, region['role'].upper() + ' ROI', polygon, color)
    for person in people:
        x1, y1, x2, y2 = map(round, person['box'])
        color = colors[person['role']]
        label = person['role'].upper()
        draw_detection(frame, person['box'], label, color, person=True)
        cv2.circle(frame, (round((x1 + x2) / 2), y2), 5, color, -1, cv2.LINE_AA)


def draw_analytics(frame, counters, rois, second, last_analysis, start, end,
                   person_rois=(), people=()):
    draw_roi_fills(frame, rois, person_rois)
    draw_scene(frame, counters, rois, second, start <= second < end)
    draw_people(frame, person_rois, people)
    summaries = {lane: counter.summary(last_analysis) for lane, counter in counters.items()}
    draw_panel(frame, summaries, second, start, end)
