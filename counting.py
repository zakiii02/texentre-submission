"""Tracking and counting helpers. User settings are in main.py."""
from types import SimpleNamespace
from motion import MotionBridge
import numpy as np

from utils import appearance, inside, iou, overlaps_bag, touches_roi


class LaneCounter:
    def __init__(self, rois, fps, start_seconds):
        self.rois = rois
        self.start = start_seconds
        self.objects = {}
        self.aliases = {}
        self.initial_conveyor = None
        self.events = []
        self.trackers = None
        self.fps = fps
        self.motion = MotionBridge()

    def track(self, detections, frame, second):
        """Separate ByteTrack instances prevent item/bag class ID switches."""
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        if self.trackers is None:
            args = SimpleNamespace(track_high_thresh=.30, track_low_thresh=.10,
                                   new_track_thresh=.35, track_buffer=round(self.fps*3),
                                   match_thresh=.85, fuse_score=False)
            self.trackers = {cls: BYTETracker(args) for cls in (0,1)}
        if self.initial_conveyor is None:
            self.initial_conveyor = int(sum(
                int(detection[5]) == 0
                and detection[4] >= .35
                and touches_roi(detection[:4], self.rois['conveyor'])
                for detection in detections
            ))
        observations = []
        for cls, tracker in self.trackers.items():
            data = np.array([detection for detection in detections if int(detection[5]) == cls], np.float32).reshape(-1,6)
            tracks = tracker.update(Boxes(data, frame.shape[:2]))
            for row in tracks:
                box = row[:4].tolist()
                observations.append(dict(raw_id=(cls,int(row[4])), cls=cls, box=box,
                                         hist=appearance(frame, box)))
        self.update(observations, second, frame)

    def update(self, observations, second, frame=None):
        """Maintain persistent identities and record each first ROI visit."""
        # Reserve existing matches before stitching new tracker IDs to dormant identities.
        reserved = {self.aliases[detection['raw_id']] for detection in observations if detection['raw_id'] in self.aliases}
        used = set()
        for detection in observations:
            key = detection['raw_id']
            identity = self.aliases.get(key)
            if identity in used:
                # Two simultaneously observed tracks cannot be one physical object.
                identity = None
            if identity is None:
                matches = []
                for previous in self.objects.values():
                    if (previous['id'] in reserved or previous['cls'] != detection['cls']
                            or not 0 < second-previous['last'] <= 3):
                        continue
                    similarity = float(np.dot(previous['hist'], detection['hist']))
                    detector_match = iou(previous['box'], detection['box']) > .30 and similarity >= .85
                    visual_match = (second-previous.get('visual_last', -100) <= .3
                                    and iou(previous.get('visual_box', previous['box']), detection['box']) > .20
                                    and similarity >= .65)
                    if detector_match or visual_match:
                        matches.append(previous)
                # Ambiguity is preferable to merging two identical neighbouring products.
                if len(matches) == 1:
                    identity = matches[0]['id']
                    self.events.append(dict(type='identity_relinked', id=identity, second=second))
                else:
                    identity = len(self.objects)+1
                    self.objects[identity] = dict(id=identity, cls=detection['cls'], box=detection['box'], hist=detection['hist'],
                        first=second, last=second, hits=0, zones=set(), contact=None,
                        first_hist=detection['hist'].copy(),
                        packed=None, movement=0., first_box=detection['box'], confirmed=False,
                        reappearance_hits=0, possible_previous_ids=[track['id'] for track in matches])
                self.aliases[key] = identity
            reserved.add(identity)
            used.add(identity)
            track = self.objects[identity]
            if track['packed']:
                track['reappearance_hits'] = track.get('reappearance_hits', 0) + 1 if second-track['last'] <= .5 else 1
                if track['reappearance_hits'] >= 3:
                    self.events.append(dict(type='packing_retracted', id=identity, second=second))
                    track['packed'] = None
                    track['reappearance_hits'] = 0
            track['movement'] += float(np.linalg.norm(np.array(track['box'][:2])-np.array(detection['box'][:2])))
            track.update(box=detection['box'], hist=detection['hist'], last=second, hits=track['hits']+1)
            track['confirmed'] = track['hits'] >= 3
            for zone, polygon in self.rois.items():
                in_zone = touches_roi(track['box'], polygon) if track['cls']==0 else inside(track['box'],polygon)
                if in_zone and zone not in track['zones']:
                    track['zones'].add(zone)
                    if zone == 'bagging':
                        track['bagging_first'] = second
                    self.events.append(dict(type='zone_entry', id=identity, zone=zone, second=second))
        if frame is not None:
            self.motion.update(frame, self.objects, second)
        self.update_packing(second)

    def update_packing(self, second):
        """Keep short-lived visual/contact evidence through detector gaps."""
        # A bag may flicker for a few frames during hand occlusion.
        bags = [track for track in self.objects.values()
                if track['confirmed'] and track['cls'] == 1
                and second-track['last'] <= 1.0
                and inside(track['box'], self.rois['bagging'])]
        items = [track for track in self.objects.values()
                 if track['confirmed'] and track['cls'] == 0 and 'bagging' in track['zones']
                 and not track.get('possible_previous_ids')]
        for item in items:
            missing = second-item['last']
            visual_recent = second-item.get('visual_last', -100) <= .3
            evidence_box = item.get('visual_box', item['box']) if visual_recent else item['box']
            contacts = [bag for bag in bags if overlaps_bag(evidence_box, bag['box'])]
            # Never clear good evidence because of one empty bag-detection frame.
            if (missing < .01 or visual_recent) and len(contacts) == 1:
                item['contact'] = dict(second=second, bag_id=contacts[0]['id'],
                                       source=item.get('visual_source', 'detector'))
            successors = self.possible_successors(item)
            item['possible_successor_ids'] = successors
            unresolved = [identity for identity in successors
                          if not self.finished_fragment(self.objects[identity], second)]
            if unresolved:
                if item['packed']:
                    self.events.append(dict(type='packing_retracted', id=item['id'],
                                            second=second, reason='possible_replacement_track'))
                    item['packed'] = None
                continue
            if item['packed'] or missing < .8 or missing > 8:
                continue
            contact = item['contact']
            if contact and second-contact['second'] <= 1.2:
                bag = next((bag for bag in bags if bag['id'] == contact['bag_id']), None)
                if bag is not None:
                    reason = ('motion_bridge_into_bag' if contact.get('source') == 'optical_flow'
                              else 'overlap_then_disappearance')
                    self.pack(item, bag, second, reason)
                    continue
            # A bag can become recognizable several seconds after a bundle was lifted.
            # Use the final detector or visual position in the bagging sequence.
            if item['movement'] >= 20:
                # A folded bag can become recognizable after a hand moves away.
                candidate_boxes = [item['box'], item.get('visual_box', item['box'])]
                nearby = []
                for bag in bags:
                    if not 0 <= bag['first']-item['last'] <= 8:
                        continue
                    bag_boxes = [bag['first_box'], bag['box']]
                    overlaps = any(
                        overlaps_bag(box, [bag_box[0], bag_box[1]-30, bag_box[2], bag_box[3]])
                        for box in candidate_boxes for bag_box in bag_boxes
                    )
                    if overlaps:
                        nearby.append(bag)
                if len(nearby) == 1:
                    self.pack(item, nearby[0], second, 'delayed_bag_replaces_item_position')
                    continue
            # Inventory evidence is weaker than observed bag contact. Only use
            # established products, not brief fragments appearing during handling.
            stable_in_bagging = item['last'] - item.get('bagging_first', second) >= 2.0
            if missing >= 2.0 and stable_in_bagging and item['hits'] >= 15:
                new_bags = [bag for bag in bags
                            if -1.0 <= bag['first'] - item['last'] <= 8.0]
                if new_bags:
                    item['packed'] = dict(
                        bag_id=None, candidate_bag_ids=[bag['id'] for bag in new_bags],
                        second=second, evidence='bagging_inventory_depletion',
                        status='inferred_not_verified', confidence='low')
                    item['reappearance_hits'] = 0
                    self.events.append(dict(type='packing_inferred', id=item['id'], **item['packed']))

    def finished_fragment(self, track, second):
        """A brief, vanished handling fragment must not block its source forever.

        Keep blocking if it leads to another ID: that product may still be visible.
        The fragment itself is too short-lived for inventory packing evidence.
        """
        return (track['last'] - track['first'] < 2.0
                and second - track['last'] >= 2.0
                and not self.possible_successors(track))

    def possible_successors(self, item):
        """Withhold packing when a nearby new ID could be the same product.

        This is an uncertainty check, not an identity merge: two similar products
        may also be exchanged under a hand occlusion.
        """
        box = np.asarray(item['box'])
        center = (box[:2] + box[2:]) / 2
        radius = 1.5 * max(box[2:] - box[:2])
        matches = []
        for other in self.objects.values():
            if (other['id'] == item['id'] or other['cls'] != 0
                    or not other['confirmed']
                    or not 0 < other['first'] - item['last'] <= 1.0):
                continue
            first_box = np.asarray(other['first_box'])
            distance = np.linalg.norm((first_box[:2] + first_box[2:]) / 2 - center)
            similarity = max(float(np.dot(item['hist'], other['hist'])),
                             float(np.dot(item['hist'], other.get('first_hist', other['hist']))))
            if distance <= radius and similarity >= .80:
                matches.append(other['id'])
        return matches

    def pack(self, item, bag, second, reason):
        """Attach an auditable packing estimate to one existing retail item."""
        item['packed'] = dict(bag_id=bag['id'], second=second, evidence=reason,
                              status='inferred_not_verified')
        item['reappearance_hits'] = 0
        self.events.append(dict(type='packing_inferred', id=item['id'], **item['packed']))

    def fragment_sources(self, track):
        """Do not count a brief ambiguous continuation as a new retail item.

        Keep genuinely new short tracks. Only withhold a track when earlier,
        established identities already account for every ROI it visited.
        A two-second independent track or simultaneous observation releases it.
        """
        if track['cls'] != 0 or track['packed'] or track['last'] - track['first'] >= 2.0:
            return []
        sources = [self.objects[identity]
                   for identity in track.get('possible_previous_ids', [])
                   if identity in self.objects]
        if any(source['last'] >= track['first'] for source in sources):
            return []
        sources = [source for source in sources
                   if source['confirmed'] and source['cls'] == 0
                   and source['last'] - source['first'] >= 2.0
                   and source['last'] < track['first']]
        covered = set().union(*(source['zones'] for source in sources))
        if sources and track['zones'] and track['zones'] <= covered:
            return [source['id'] for source in sources]
        return []

    def summary(self, second):
        """Report per-ROI counts and bag estimates."""
        confirmed = [track for track in self.objects.values() if track['confirmed']]
        fragments = [track['id'] for track in confirmed if self.fragment_sources(track)]
        items = [track for track in confirmed
                 if track['cls'] == 0 and track['zones'] and track['id'] not in fragments]
        bags = [track for track in confirmed if track['cls'] == 1 and 'bagging' in track['zones']]
        stage_ids = {
            zone: [track['id'] for track in items if zone in track['zones']]
            for zone in self.rois
        }
        packed_ids = [track['id'] for track in items if track['packed']]

        bags_present = sum(
            second - track['last'] <= 0.4
            and inside(track['box'], self.rois['bagging'])
            for track in bags
        )
        missing_ids = [
            track['id'] for track in items
            if 'bagging' in track['zones']
            and second - track['last'] > 0.8
            and not track['packed']
        ]
        return {
            'conveyor_at_start': self.initial_conveyor,
            'conveyor_unique': len(stage_ids['conveyor']),
            'scanner_unique': len(stage_ids['scanner']),
            'bagging_area_unique': len(stage_ids['bagging']),
            'bagged_items_estimate': len(packed_ids) if packed_ids else None,
            'bags_used_unique': len(bags),
            'bags_present_at_end': bags_present,
            'retail_ids': [track['id'] for track in items],
            'excluded_fragment_ids': fragments,
            'bagged_ids': packed_ids,
            'missing_without_packing_evidence': missing_ids,
            'zone_ids': stage_ids,
        }

    def export_tracks(self):
        return [{k:(sorted(v) if isinstance(v,set) else v) for k,v in track.items() if k not in ('hist', 'first_hist')}
                for track in self.objects.values()]
