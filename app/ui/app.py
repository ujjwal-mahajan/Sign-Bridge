"""SignBridge Main Live Application (Agent 4 - UI & Integration).

Integrates:
- Agent 1: Video Capture & Keypoint Pipeline (Contracts A & B)
- Agent 2: Gesture Classification & Model Inference (Contract C)
- Agent 3: Temporal Stability Filtering, Sentence Smoothing & Async TTS (Contracts D, E, F)
- Agent 4: Live Dashboard Overlay, Interactive Controls & Contract G State Management
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import time
import cv2
import numpy as np

# Agent 1
from app.capture.camera import CameraCapture
from app.keypoints.extractor import HandKeypointExtractor
from app.keypoints.preprocessing import KeypointPreprocessor
from app.keypoints.buffer import SequenceBuffer

# Agent 2
from app.recognition.inference import ModelInference
from app.recognition.labels import VOCABULARY

# Agent 3
from app.sentence.filter import PredictionFilter
from app.sentence.builder import SentenceBuilder
from app.tts.speech import TextToSpeech

# Agent 4
from app.ui.overlay import UIOverlayRenderer


class SignBridgeApp:
    """Main SignBridge application controller and end-to-end integration layer."""

    def __init__(
        self,
        mock_camera: bool = False,
        model_path: str = "models/sign_model.pth",
        labels_path: str = "models/labels.json",
        confidence_threshold: float = 0.60,
        stability_window: int = 15,
        tts_enabled: bool = True,
        tts_rate: int = 120,
        headless: bool = False,
        target_fps: int = 30,
        fetch_gap: float = 0.5,
    ):
        """Initialize all pipeline subsystems.

        Args:
            mock_camera: Whether to use simulated frames instead of live webcam.
            model_path: Path to PyTorch model weights (.pth).
            labels_path: Path to label mapping JSON file.
            confidence_threshold: Threshold to accept candidate signs (Contract D).
            stability_window: Consecutive agreeing frames required (Contract D).
            tts_enabled: Whether Text-to-Speech audio is enabled.
            tts_rate: Speech rate in words per minute.
            headless: If True, disables cv2.imshow for CI / automated tests.
            target_fps: Target frame rate for smooth UI pacing (default 30).
            fetch_gap: Appropriate timing gap in seconds between two signs to fetch (default 0.5).
        """
        self.mock_camera = mock_camera
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.headless = headless
        self.target_fps = target_fps
        self.fetch_gap = fetch_gap

        # Subsystems
        self.camera: Optional[CameraCapture] = None
        self.extractor = HandKeypointExtractor()
        self.preprocessor = KeypointPreprocessor()
        self.seq_buffer = SequenceBuffer(sequence_length=30, feature_dim=126)

        self.model = ModelInference(
            model_path=model_path,
            labels_path=labels_path,
            confidence_threshold=confidence_threshold,
        )
        self.pred_filter = PredictionFilter(
            confidence_threshold=confidence_threshold,
            stability_window=stability_window,
            timing_gap=fetch_gap,
        )
        self.sentence_builder = SentenceBuilder()
        self.tts = TextToSpeech(rate=tts_rate, enabled=tts_enabled)
        self.renderer = UIOverlayRenderer()

        # Operational State
        self.is_running = False
        self.current_sign: Optional[str] = None
        self.current_confidence: float = 0.0
        self.last_fetch_time: Optional[float] = None
        self.last_fetched_sign: Optional[str] = None
        self.is_fetch_mode: bool = True
        self.toast_message: Optional[str] = None
        self.toast_expires_at: float = 0.0
        self.error_message: Optional[str] = None
        
        # Performance Tracking
        self.fps: float = 0.0
        self.prev_frame_time: float = time.time()
        self.frame_count: int = 0

        # Demo signs mapping for quick simulation keys (1-5)
        self.demo_shortcuts = {
            ord('1'): "hello",
            ord('2'): "thank_you",
            ord('3'): "please",
            ord('4'): "water",
            ord('5'): "help",
        }

        # Initialize Camera
        self._init_camera(mock_camera)

    def _init_camera(self, mock: bool) -> None:
        """Safely initialize or switch the camera subsystem."""
        if self.camera is not None:
            self.camera.release()

        self.mock_camera = mock
        try:
            self.camera = CameraCapture(mock=mock)
            self.camera.start()
            if not self.camera.is_opened():
                print(f"[SignBridge Warning] Camera (mock={mock}) failed to open. Switching to mock.")
                self.mock_camera = True
                self.camera = CameraCapture(mock=True)
                self.camera.start()
        except Exception as e:
            print(f"[SignBridge Error] Camera initialization exception: {e}. Using mock.")
            self.mock_camera = True
            self.camera = CameraCapture(mock=True)
            self.camera.start()

    def set_toast(self, message: str, duration: float = 2.5) -> None:
        """Display a transient notification on the UI dashboard."""
        self.toast_message = message
        self.toast_expires_at = time.time() + duration

    def get_ui_state(self) -> Dict[str, Any]:
        """Generate current Contract G state object.

        Returns:
            dict conforming to Contract G schema.
        """
        camera_ok = self.camera.is_opened() if self.camera else False
        return {
            "current_sign": self.current_sign,
            "confidence": round(float(self.current_confidence), 4),
            "recognized_words": self.sentence_builder.get_words(),
            "sentence": self.sentence_builder.get_sentence(),
            "is_fetch_mode": self.is_fetch_mode,
            "status": {
                "camera": camera_ok,
                "model": True,
                "speech": self.tts.is_available(),
                "fetch_mode": self.is_fetch_mode,
            },
        }

    def inject_prediction(
        self,
        label: str,
        confidence: float = 0.90,
        timestamp: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Simulate an inference prediction for demo or automated testing.

        Args:
            label: Sign vocabulary label string.
            confidence: Prediction confidence score in [0.0, 1.0].
            timestamp: Optional explicit Unix timestamp.

        Returns:
            Contract D event if stabilized, otherwise None.
        """
        now = timestamp if timestamp is not None else time.time()
        # For programmatic simulation/tests without explicit timestamps,
        # advance simulated time beyond the timing gap when switching to a different sign
        if timestamp is None and self.last_fetch_time is not None:
            if label != self.last_fetched_sign:
                gap_target = self.last_fetch_time + self.fetch_gap + 0.05
                if now < gap_target:
                    now = gap_target

        pred = {
            "label": label,
            "confidence": confidence,
            "timestamp": now,
        }
        self.current_sign = label
        self.current_confidence = confidence

        # Check timing gap
        time_since_fetch = (now - self.last_fetch_time) if self.last_fetch_time is not None else float("inf")
        if time_since_fetch < self.fetch_gap:
            self.is_fetch_mode = False
            self.pred_filter.process_prediction(pred)
            return None

        self.is_fetch_mode = True
        stable_event = self.pred_filter.process_prediction(pred)
        if stable_event:
            self.last_fetch_time = now
            self.last_fetched_sign = stable_event["word"]
            self.sentence_builder.add_word(stable_event)
            self.set_toast(f"Accepted: {stable_event['word']}")
        return stable_event

    def step(self) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """Execute one complete frame through the 4-agent pipeline.

        Returns:
            Tuple of:
            - canvas: Complete composite dashboard image (np.ndarray).
            - ui_state: Current Contract G state dictionary.
            - should_continue: False if pipeline encountered termination.
        """
        # Calculate FPS
        now = time.time()
        dt = now - self.prev_frame_time
        self.prev_frame_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps

        # Clear expired toast
        if self.toast_message and now > self.toast_expires_at:
            self.toast_message = None

        # Stage 1: Capture & Preprocessing (Agent 1)
        raw_frame = self.camera.get_frame() if self.camera else None
        annotated_frame = raw_frame

        if raw_frame is not None:
            # Extract keypoints and draw skeleton if available
            raw_keypoints, info, annotated_frame = self.extractor.extract_keypoints(raw_frame, draw=True)
            norm_keypoints = self.preprocessor.normalize(raw_keypoints)
            self.seq_buffer.add_frame(norm_keypoints)

        # Stage 2: Temporal Inference (Agent 2)
        if raw_frame is not None and info.get("hands_count", 0) == 0:
            self.current_sign = None
            self.current_confidence = 0.0
            self.pred_filter.reset()
        elif self.seq_buffer.is_ready():
            try:
                seq = self.seq_buffer.get_sequence() # Shape: (30, 126)
                prediction = self.model.predict(seq) # Contract C
                pred_label = prediction["label"]
                pred_conf = prediction["confidence"]

                # Always update the displayed sign and confidence on the UI
                self.current_sign = pred_label
                self.current_confidence = pred_conf

                # Check timing gap between two signs to fetch
                time_since_fetch = (now - self.last_fetch_time) if self.last_fetch_time is not None else float("inf")
                in_gap = time_since_fetch < self.fetch_gap

                if in_gap:
                    # Timing gap active:
                    # 1. If same sign as last fetched: do not enter fetch mode
                    # 2. If different sign: display it only and nothing else!
                    self.is_fetch_mode = False
                    # Still feed prediction into filter so it tracks consecutive frames
                    self.pred_filter.process_prediction(prediction)
                else:
                    # Timing gap elapsed: fetch mode is active
                    self.is_fetch_mode = True
                    # Stage 3: Temporal Stability & Sentence Builder (Agent 3)
                    stable_event = self.pred_filter.process_prediction(prediction) # Contract D
                    if stable_event:
                        self.last_fetch_time = now
                        self.last_fetched_sign = stable_event["word"]
                        self.sentence_builder.add_word(stable_event) # Contract E
                        self.set_toast(f"Recognized: {stable_event['word']}")
            except Exception as e:
                self.error_message = f"Inference Error: {e}"

        # Get diagnostic stats
        filter_status = self.pred_filter.get_status()
        stability_count = filter_status.get("consecutive_count", 0)

        # Stage 4: Contract G State & Render Dashboard (Agent 4)
        ui_state = self.get_ui_state()
        is_speaking = self.tts.is_busy()

        canvas = self.renderer.render(
            camera_frame=annotated_frame,
            ui_state=ui_state,
            fps=self.fps,
            toast_message=self.toast_message,
            stability_count=stability_count,
            stability_target=self.stability_window,
            is_mock_camera=self.mock_camera,
            is_speaking=is_speaking,
            error_message=self.error_message,
        )

        return canvas, ui_state, True

    def handle_key(self, key_code: int) -> bool:
        """Process keyboard interactions.

        Args:
            key_code: ASCII/OpenCV keycode (from cv2.waitKey).

        Returns:
            bool: True if application should keep running; False to quit.
        """
        if key_code == -1:
            return True

        key = key_code & 0xFF

        # [Q] or [ESC]: Quit
        if key == ord('q') or key == 27:
            return False

        # [S]: Speak current sentence
        elif key == ord('s'):
            sentence_data = self.sentence_builder.build_sentence()
            if sentence_data["text"]:
                self.tts.speak_sentence(sentence_data)
                self.set_toast("Speaking Sentence...")
            else:
                self.set_toast("Buffer is empty!")

        # [C]: Clear sentence buffer and reset filter
        elif key == ord('c'):
            self.sentence_builder.clear()
            self.pred_filter.reset()
            self.set_toast("Cleared Word Buffer")

        # [Backspace] or [B]: Delete last word
        elif key in (ord('\b'), 8, 127, ord('b')):
            removed = self.sentence_builder.remove_last_word()
            if removed:
                self.set_toast(f"Removed: {removed}")
            else:
                self.set_toast("Buffer already empty")

        # [M]: Toggle between live camera and mock camera
        elif key == ord('m'):
            new_mode = not self.mock_camera
            self._init_camera(new_mode)
            self.set_toast("Switched to MOCK Camera" if new_mode else "Switched to LIVE Camera")

        # [E]: Export Transcript (Hackathon Feature)
        elif key == ord('e'):
            words = self.sentence_builder.buffer
            if not words:
                self.set_toast("Nothing to export!")
            else:
                import time, os
                os.makedirs("exports", exist_ok=True)
                filename = f"exports/transcript_{int(time.time())}.txt"
                with open(filename, "w") as f:
                    f.write("SignBridge Diagnostic Export\n")
                    f.write("==============================\n")
                    f.write(f"Timestamp: {time.ctime()}\n")
                    f.write("Translated Sentence:\n")
                    f.write(" ".join(words).upper() + "\n")
                self.set_toast(f"Exported to {filename}")

        # [1-5]: Quick sign demo injection
        elif key in self.demo_shortcuts:
            sign_word = self.demo_shortcuts[key]
            # Emulate 5 agreeing frames to pass stability filter
            for _ in range(self.stability_window):
                self.inject_prediction(sign_word, confidence=0.92)
            self.set_toast(f"Demo Injected: {sign_word}")

        return True

    def reset(self) -> None:
        """Reset all sentence, filter, and sequence states."""
        self.pred_filter.reset()
        self.sentence_builder.clear()
        self.seq_buffer.reset()
        self.current_sign = None
        self.current_confidence = 0.0
        self.last_fetch_time = None
        self.last_fetched_sign = None
        self.is_fetch_mode = True
        self.toast_message = None
        self.error_message = None

    def run(self, max_frames: Optional[int] = None) -> None:
        """Run the interactive UI loop.

        Args:
            max_frames: Optional maximum frames to execute (useful for benchmarks/tests).
        """
        self.is_running = True
        window_name = "SignBridge — Live Translator (Agent 4)"

        if not self.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        target_frame_time = 1.0 / max(1, self.target_fps)
        frames_run = 0
        try:
            while self.is_running:
                frame_start_time = time.perf_counter()
                canvas, ui_state, should_continue = self.step()
                if not should_continue:
                    break

                frames_run += 1
                if max_frames and frames_run >= max_frames:
                    break

                if not self.headless:
                    cv2.imshow(window_name, canvas)
                    elapsed = time.perf_counter() - frame_start_time
                    remaining_ms = int((target_frame_time - elapsed) * 1000)
                    wait_ms = max(1, remaining_ms)
                    key = cv2.waitKey(wait_ms)
                    if not self.handle_key(key):
                        break
                else:
                    # In headless mode, sleep for remaining frame budget
                    elapsed = time.perf_counter() - frame_start_time
                    sleep_time = max(0.001, target_frame_time - elapsed)
                    time.sleep(sleep_time)

        finally:
            self.close()

    def close(self) -> None:
        """Cleanly release resources."""
        self.is_running = False
        if self.camera:
            self.camera.release()
        if self.tts:
            self.tts.shutdown()
        if not self.headless:
            cv2.destroyAllWindows()


def run_app():
    """CLI Entry point for running the application."""
    import argparse

    parser = argparse.ArgumentParser(description="SignBridge Real-Time Sign Language Translator")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock camera mode")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio speech synthesis")
    parser.add_argument("--frames", type=int, default=None, help="Limit total run frames")
    args = parser.parse_args()

    print("============================================================")
    print("   SignBridge — Real-Time Sign Language Translator (M4)     ")
    print("============================================================")
    print(f"[Config] Mock Camera: {args.mock}")
    print(f"[Config] Headless: {args.headless}")
    print(f"[Config] TTS Enabled: {not args.no_tts}")
    print("Controls: [S] Speak | [C] Clear | [E] Export | [Backspace] Undo | [M] Toggle Mock | [1-5] Signs | [Q] Quit")
    print("------------------------------------------------------------")

    app = SignBridgeApp(
        mock_camera=args.mock,
        tts_enabled=not args.no_tts,
        headless=args.headless,
    )
    app.run(max_frames=args.frames)


if __name__ == "__main__":
    run_app()
