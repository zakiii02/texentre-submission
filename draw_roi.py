import cv2
import numpy as np
import os
import sys
import yaml

# --------------------------------------------------
# Constants
# --------------------------------------------------
ROI_CONFIG_FILE = "roi_config.yml"
IMAGE_OUTPUT_DIR = "Roi_Annotated_Images"


def load_roi_config(config_path=ROI_CONFIG_FILE):
    """Load existing ROI config or create a new empty file if not found."""
    if not os.path.exists(config_path):
        try:
            with open(config_path, "w") as f:
                yaml.safe_dump({}, f)
            print(f"Created new ROI config file: {config_path}")
        except Exception as e:
            print(f"Error creating {config_path}: {e}")
        return {}

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
        return {}


# --------------------------------------------------
# YAML Flow Formatting Helper
# --------------------------------------------------
class FlowSeq(list):
    """Enables clean [x, y] flow formatting inside YAML lists."""
    pass


def _flow_seq_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowSeq, _flow_seq_representer, Dumper=yaml.SafeDumper)


def save_roi_config(rois, config_path=ROI_CONFIG_FILE):
    """Save all ROIs into YAML with clean [x, y] coordinates format."""
    formatted = {}
    for name, pts in rois.items():
        formatted[name] = [FlowSeq([int(p[0]), int(p[1])]) for p in pts]

    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(formatted, f, sort_keys=False)
        print(f"Updated ROI config successfully: {config_path}")
        return True
    except Exception as e:
        print(f"Error saving {config_path}: {e}")
        return False


class ROIAnnotator:
    def __init__(self, video_path=None, config_path=ROI_CONFIG_FILE):
        self.config_path = config_path
        self.saved_rois = load_roi_config(self.config_path)

        if self.saved_rois:
            print(f"\nLoaded {len(self.saved_rois)} existing ROI(s) from {self.config_path}:")
            for name, pts in self.saved_rois.items():
                print(f"  - {name}: {len(pts)} points")
        else:
            print(f"\nNo existing ROIs in {self.config_path}. Ready to add new ones.")

        self.video_path = video_path
        self.window_name = "ROI Annotation"
        self.frozen = False
        self.original_frame = None
        self.display_frame = None
        self.points = []
        self.redo_stack = []

    def _open_video(self):
        if not self.video_path:
            default_video = "videoplayback.mp4" if os.path.isfile("videoplayback.mp4") else ""
            prompt_str = f"Enter video path or RTSP URL [{default_video}]: " if default_video else "Enter video path or RTSP URL: "
            user_input = input(prompt_str).strip()
            self.video_path = user_input if user_input else default_video

        if not self.video_path:
            print("Invalid video path.")
            raise SystemExit(1)

        if "://" not in self.video_path and not os.path.isfile(self.video_path):
            print(f"Video file does not exist: {self.video_path}")
            raise SystemExit(1)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            raise SystemExit(1)

        return cap

    def _update_display(self):
        if self.original_frame is None:
            return

        self.display_frame = self.original_frame.copy()

        # 1. Draw saved ROIs from config
        for roi_name, roi_pts in self.saved_rois.items():
            if len(roi_pts) >= 3:
                pts_arr = np.array(roi_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.display_frame, [pts_arr], True, (255, 200, 0), 2)
                label_x, label_y = roi_pts[0][0], roi_pts[0][1]
                cv2.putText(
                    self.display_frame,
                    roi_name,
                    (label_x + 5, label_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

        # 2. Draw active points being marked
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                self.display_frame,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Draw connecting lines
        if len(self.points) >= 2:
            pts = np.array(self.points, np.int32)
            cv2.polylines(self.display_frame, [pts], False, (0, 255, 255), 2)

        # Close polygon when at least 3 points
        if len(self.points) >= 3:
            pts = np.array(self.points, np.int32)
            cv2.polylines(self.display_frame, [pts], True, (0, 0, 255), 2)

        cv2.imshow(self.window_name, self.display_frame)

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.frozen:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.redo_stack.clear()
            print(f"Added point {len(self.points)}: ({x}, {y})")
            self._update_display()

    def _save_active_roi(self):
        if len(self.points) < 3:
            print("At least 3 ROI points are required to save.")
            return False

        # Close OpenCV window so terminal input is unblocked
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print("\nFinal ROI Points:")
        print([[x, y] for x, y in self.points])

        roi_name = input(
            "\nEnter ROI name (e.g. CONVEYOR, SCANNER, BAGGING_AREA, CUSTOMER_SIDE, CONVEYOR_2): "
        ).strip()

        if not roi_name:
            print("Invalid name. ROI not saved.")
            return False

        if roi_name in self.saved_rois:
            confirm = input(
                f"'{roi_name}' already exists in {self.config_path}. Overwrite? (y/n, default: y): "
            ).strip().lower()
            if confirm in ('n', 'no'):
                print("Save cancelled.")
                return False

        # Update saved ROIs and save to YAML
        self.saved_rois[roi_name] = [[int(x), int(y)] for x, y in self.points]
        save_roi_config(self.saved_rois, self.config_path)

        # Ensure output directory exists
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

        # Save single ROI image
        filename = os.path.join(IMAGE_OUTPUT_DIR, f"{roi_name}.jpg")
        final_image = self.original_frame.copy()
        pts_arr = np.array(self.points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(final_image, [pts_arr], True, (0, 0, 255), 2)
        for x, y in self.points:
            cv2.circle(final_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            final_image,
            roi_name,
            (self.points[0][0] + 8, self.points[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        cv2.imwrite(filename, final_image)
        print(f"Saved annotated image: {filename}")

        # Save combined overview image
        all_filename = os.path.join(IMAGE_OUTPUT_DIR, "all_rois_annotated.jpg")
        all_image = self.original_frame.copy()
        for name, r_pts in self.saved_rois.items():
            arr = np.array(r_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(all_image, [arr], True, (0, 255, 255), 2)
            cv2.putText(
                all_image,
                name,
                (r_pts[0][0] + 5, r_pts[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        cv2.imwrite(all_filename, all_image)
        print(f"Saved combined overview image: {all_filename}")

        return True

    def _setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def run(self):
        cap = self._open_video()
        self._setup_window()

        print("\n========== ROI ANNOTATION ==========")
        print("F     : Freeze / Unfreeze frame")
        print("Click : Add ROI point")
        print("Z     : Undo point")
        print("Y     : Redo point")
        print("C     : Clear active points")
        print("S     : Save current ROI & continue next ROI")
        print("Q     : Save current ROI and quit (or quit if no points drawn)")
        print("ESC   : Exit without saving active points")
        print("====================================\n")

        try:
            while True:
                if not self.frozen:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video or unable to read frame.")
                        break

                    self.display_frame = frame.copy()
                    for roi_name, roi_pts in self.saved_rois.items():
                        if len(roi_pts) >= 3:
                            pts_arr = np.array(roi_pts, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(self.display_frame, [pts_arr], True, (255, 200, 0), 2)
                            cv2.putText(
                                self.display_frame,
                                roi_name,
                                (roi_pts[0][0] + 5, roi_pts[0][1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 255, 0),
                                2
                            )

                    cv2.imshow(self.window_name, self.display_frame)

                key = cv2.waitKey(30) & 0xFF

                # Freeze / unfreeze frame
                if key == ord('f'):
                    self.frozen = not self.frozen
                    if self.frozen:
                        self.original_frame = frame.copy()
                        print("\nFrame frozen. Click to draw ROI points.")
                        self._update_display()
                    else:
                        print("\nFrame unfrozen. Video playing.")

                # Undo point
                elif key == ord('z') and self.frozen:
                    if self.points:
                        removed = self.points.pop()
                        self.redo_stack.append(removed)
                        print(f"Undo point: {removed}")
                        self._update_display()
                    else:
                        print("No points to undo.")

                # Redo point
                elif key == ord('y') and self.frozen:
                    if self.redo_stack:
                        restored = self.redo_stack.pop()
                        self.points.append(restored)
                        print(f"Redo point: {restored}")
                        self._update_display()
                    else:
                        print("No points to redo.")

                # Clear points of current ROI
                elif key == ord('c') and self.frozen:
                    self.points.clear()
                    self.redo_stack.clear()
                    print("Cleared active points.")
                    self._update_display()

                # Save current ROI & continue next ROI
                elif key == ord('s'):
                    if not self.frozen:
                        print("Freeze a frame first using F.")
                        continue

                    if len(self.points) < 3:
                        print("At least 3 ROI points are required to save.")
                        continue

                    saved_ok = self._save_active_roi()
                    if saved_ok:
                        self.points.clear()
                        self.redo_stack.clear()
                        self._setup_window()
                        print("\nReady to draw the next ROI! Click points, or press Q to exit when done.")
                        self._update_display()

                # Save and quit
                elif key == ord('q'):
                    if self.frozen and len(self.points) >= 3:
                        self._save_active_roi()
                    print("Exiting application.")
                    break

                # Exit without saving
                elif key == 27:  # ESC
                    print("Exiting without saving active points.")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Resources released. Program exited.")


if __name__ == "__main__":
    video_arg = sys.argv[1] if len(sys.argv) > 1 else None
    annotator = ROIAnnotator(video_path=video_arg)
    annotator.run()
