"""
benchmark_extractor.py
Streamlit app for extracting and editing benchmark reps (Golden Reps).
Uses MediaPipe BlazePose to extract 33 3D landmarks, normalizes, smooths,
and allows interactive editing of start/end frames, landmarks, and checkpoints.
"""

import json
import hashlib
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import streamlit as st
from pathlib import Path
from scipy.signal import savgol_filter
import subprocess
import tempfile
import os

# ================= CONSTANTS =================
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_SHOULDER, R_SHOULDER = 11, 12

POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
])

# ================= HELPERS =================
def download_youtube_video(url, output_path="temp_video.mp4"):
    """Download a YouTube video using yt-dlp."""
    try:
        with st.spinner("Downloading YouTube video..."):
            subprocess.run([
                "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "-o", output_path, "--merge-output-format", "mp4", url
            ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to download video: {e.stderr.decode()}")
        return None

def get_video_codec(video_path):
    """Use ffprobe to detect video codec name."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return None


def convert_video_if_needed(video_path, start_time=None, duration=None):
    """Convert video to H.264/AAC MP4 using ffmpeg when needed."""
    codec = get_video_codec(video_path)

    # If already H.264/AAC in MP4 container, try to read directly
    if codec in ("h264", "avc1"):
        cap = cv2.VideoCapture(str(video_path))
        ret, _ = cap.read()
        cap.release()
        if ret:
            return video_path

    with st.spinner("Converting video to H.264/AAC MP4..."):
        temp_dir = tempfile.mkdtemp()
        converted = os.path.join(temp_dir, "converted.mp4")

        cmd = ["ffmpeg", "-y", "-i", str(video_path)]

        if start_time is not None:
            cmd.extend(["-ss", f"{start_time:.2f}"])

        cmd.extend([
            "-t", f"{duration if duration else 10:.2f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-tune", "fastdecode",
            "-c:a", "aac", "-ac", "2",
            "-movflags", "+faststart",
            converted
        ])

        subprocess.run(cmd, capture_output=True, timeout=120, check=False)

        if os.path.exists(converted):
            st.success("Video converted and ready!")
            return converted
        else:
            st.warning("Could not convert video")
            return video_path

def get_pose_detector():
    """Initialize and return a PoseLandmarker instance."""
    model_path = Path("pose_landmarker.task")

    if not model_path.exists():
        with st.spinner("Downloading pose model..."):
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            subprocess.run(["wget", "-O", str(model_path), url], check=True)
            st.success("Model downloaded!")

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
    )

    return vision.PoseLandmarker.create_from_options(options)

def get_video_info_ffprobe(video_path):
    """Get video metadata using ffprobe as fallback."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            fps_str = lines[0]
            fps = eval(fps_str) if '/' in fps_str else float(fps_str)
            total_frames = int(lines[1]) if lines[1].isdigit() else 0
            return fps, total_frames
    except Exception:
        pass
    return 30, 0


def get_video_info(video_path):
    """Get video metadata, falling back to ffprobe if cv2 fails."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Fallback to ffprobe if frame count is zero or cv2 fails
    if total_frames == 0:
        fps_ff, total_frames_ff = get_video_info_ffprobe(video_path)
        if total_frames_ff > 0:
            fps, total_frames = fps_ff, total_frames_ff

    return fps, total_frames

def extract_landmarks(video_path, start_frame=0, end_frame=None, progress_bar=None, roi=None):
    """Extract 33 3D landmarks from each frame using MediaPipe Pose Landmarker."""
    detector = get_pose_detector()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames - 1

    landmarks_data = []
    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for idx, frame_num in enumerate(range(start_frame, end_frame + 1)):
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to ROI if specified
        if roi:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]

        frames.append(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_num * 1000 / fps)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            frame_landmarks = []
            for lm in result.pose_landmarks[0]:
                # Adjust landmarks to original coordinates if ROI used
                if roi:
                    lm.x = (lm.x * w) + x
                    lm.y = (lm.y * h) + y
                frame_landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
            landmarks_data.append(frame_landmarks)
        else:
            landmarks_data.append([[0,0,0,0] for _ in range(33)])

        if progress_bar is not None:
            progress_bar.progress((idx + 1) / (end_frame - start_frame + 1))

    cap.release()
    detector.close()

    if len(landmarks_data) == 0:
        st.error("No frames extracted from video")
        return None, frames

    arr = np.array(landmarks_data, dtype=np.float32)
    if len(arr.shape) != 3 or arr.shape[1] != 33 or arr.shape[2] != 4:
        st.error(f"Unexpected landmarks shape: {arr.shape}")
        return None, frames

    return arr, frames

def normalize_skeleton(landmarks):
    """Normalize skeleton by translating hip midpoint to (0,0,0)."""
    normalized = landmarks.copy()
    for i in range(len(landmarks)):
        hip_mid = (landmarks[i][L_HIP][:3] + landmarks[i][R_HIP][:3]) / 2
        normalized[i][:, :3] -= hip_mid
    return normalized

def smooth_landmarks(landmarks, window=7, poly=2):
    """Apply Savitzky-Golay filter to smooth landmarks."""
    if len(landmarks.shape) != 3:
        st.error(f"Expected 3D array, got shape {landmarks.shape}")
        return landmarks

    n_frames, n_lm, n_dim = landmarks.shape
    smoothed = np.zeros_like(landmarks)

    for lm_idx in range(n_lm):
        for dim in range(3):
            if n_frames >= window:
                smoothed[:, lm_idx, dim] = savgol_filter(
                    landmarks[:, lm_idx, dim], window, poly
                )
            else:
                smoothed[:, lm_idx, dim] = landmarks[:, lm_idx, dim]

    smoothed[:, :, 3] = landmarks[:, :, 3]
    return smoothed

def calculate_knee_angle(landmarks, frame_idx):
    """Calculate average knee angle for a given frame."""
    def angle_between(a, b, c):
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
        cos = np.clip(np.dot(ba, bc) / denom, -1, 1)
        return np.degrees(np.arccos(cos))

    l_hip = landmarks[frame_idx][L_HIP][:3]
    l_knee = landmarks[frame_idx][L_KNEE][:3]
    l_ankle = landmarks[frame_idx][L_ANKLE][:3]
    r_hip = landmarks[frame_idx][R_HIP][:3]
    r_knee = landmarks[frame_idx][R_KNEE][:3]
    r_ankle = landmarks[frame_idx][R_ANKLE][:3]

    l_angle = angle_between(l_hip, l_knee, l_ankle)
    r_angle = angle_between(r_hip, r_knee, r_ankle)
    return (l_angle + r_angle) / 2

def detect_checkpoints(landmarks, start_frame, end_frame):
    """Detect 3 checkpoints: Start, Bottom, Lockout based on knee angles."""
    knee_angles = []
    for i in range(start_frame, min(end_frame + 1, len(landmarks))):
        knee_angles.append((i, calculate_knee_angle(landmarks, i)))

    if not knee_angles:
        return {'start': start_frame, 'bottom': start_frame, 'lockout': end_frame}

    angles = [a[1] for a in knee_angles]

    start = knee_angles[0][0]
    min_idx = np.argmin(angles)
    bottom = knee_angles[min_idx][0]
    lockout = knee_angles[-1][0]

    return {'start': int(start), 'bottom': int(bottom), 'lockout': int(lockout)}

def draw_skeleton_on_frame(frame, landmarks_frame, width=None, height=None):
    """Draw pose landmarks on a frame using cv2 directly."""
    annotated = frame.copy()
    h, w = frame.shape[:2]
    if height is not None:
        h = height
    if width is not None:
        w = width

    connections = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(24,26),
        (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
        (27,31),(28,32)
    ]

    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks_frame) and end_idx < len(landmarks_frame):
            lm1, lm2 = landmarks_frame[start_idx], landmarks_frame[end_idx]

            if lm1[3] > 0.5 and lm2[3] > 0.5:
                # Landmarks are in 0-1 range from MediaPipe
                x1, y1 = int(lm1[0] * w), int(lm1[1] * h)
                x2, y2 = int(lm2[0] * w), int(lm2[1] * h)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmarks
    for lm in landmarks_frame:
        if lm[3] > 0.5:
            x, y = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

    return annotated

def render_skeleton_video(skeleton_data, output_video_path, width=640, height=360, fps=30):
    """Render a video with only the skeleton overlay (no original video)."""
    COLOR_POINT = (0, 255, 0)
    COLOR_LINE = (0, 255, 255)

    temp_output = output_video_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    for frame_landmarks in skeleton_data:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        points = {}
        for idx, lm in enumerate(frame_landmarks):
            if lm[3] > 0.5:
                points[idx] = (int(lm[0] * width), int(lm[1] * height))

        for idx, p in points.items():
            cv2.circle(img, p, 4, COLOR_POINT, -1)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in points and end_idx in points:
                cv2.line(img, points[start_idx], points[end_idx], COLOR_LINE, 2)

        out.write(img)
    out.release()

    subprocess.run(['ffmpeg', '-y', '-i', temp_output, '-vcodec', 'libx264',
                   '-pix_fmt', 'yuv420p', '-f', 'mp4', output_video_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_output):
        os.remove(temp_output)
    return output_video_path

# ================= STREAMLIT APP =================
def main():
    st.set_page_config(page_title="Benchmark Rep Extractor", layout="wide")
    st.title("Benchmark Rep Extractor (Golden Reps)")

    # Initialize session state
    if 'landmarks' not in st.session_state:
        st.session_state.landmarks = None
    if 'normalized' not in st.session_state:
        st.session_state.normalized = None
    if 'smoothed' not in st.session_state:
        st.session_state.smoothed = None
    if 'smoothed_display' not in st.session_state:
        st.session_state.smoothed_display = None
    if 'frames' not in st.session_state:
        st.session_state.frames = None
    if 'start_frame' not in st.session_state:
        st.session_state.start_frame = 0
    if 'end_frame' not in st.session_state:
        st.session_state.end_frame = 0
    if 'checkpoints' not in st.session_state:
        st.session_state.checkpoints = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'converted_path' not in st.session_state:
        st.session_state.converted_path = None
    if 'fps' not in st.session_state:
        st.session_state.fps = 30
    if 'total_frames' not in st.session_state:
        st.session_state.total_frames = 0

    # Landmark names for human-readable labels
    LANDMARK_NAMES = {
        0: "Nose", 1: "Left Eye Inner", 2: "Left Eye",
        3: "Left Eye Outer", 4: "Right Eye Inner", 5: "Right Eye",
        6: "Right Eye Outer", 7: "Left Ear", 8: "Right Ear",
        9: "Mouth Left", 10: "Mouth Right",
        11: "Left Shoulder", 12: "Right Shoulder",
        13: "Left Elbow", 14: "Right Elbow",
        15: "Left Wrist", 16: "Right Wrist",
        17: "Left Pinky", 18: "Right Pinky",
        19: "Left Index", 20: "Right Index",
        21: "Left Thumb", 22: "Right Thumb",
        23: "Left Hip", 24: "Right Hip",
        25: "Left Knee", 26: "Right Knee",
        27: "Left Ankle", 28: "Right Ankle",
        29: "Left Heel", 30: "Right Heel",
        31: "Left Foot Index", 32: "Right Foot Index"
    }

    # ================= MAIN UI =================

    # Video Input Section
    st.header("Video Input")
    col1, col2 = st.columns(2)

    with col1:
        input_method = st.radio("Input method", ["Upload Video", "YouTube URL"])

        if input_method == "Upload Video":
            video_file = st.file_uploader("Upload video file", type=None)
            if video_file:
                # Save uploaded file to a temporary file with original extension
                suffix = Path(video_file.name).suffix or '.tmp'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.read())
                    st.session_state.video_path = tmp.name

        else:
            youtube_url = st.text_input("YouTube URL")
            if youtube_url and st.button("Fetch from YouTube"):
                temp_path = "temp_video.mp4"
                result = download_youtube_video(youtube_url, temp_path)
                if result:
                    st.session_state.video_path = result
                    st.success("Video downloaded!")

    # Convert and get video info
    if st.session_state.video_path and st.session_state.converted_path is None:
        # Get duration for conversion
        cap_temp = cv2.VideoCapture(st.session_state.video_path)
        fps_temp = cap_temp.get(cv2.CAP_PROP_FPS) or 30
        total_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        duration_sec_temp = total_temp / fps_temp

        # Pass start_time and duration for on-demand conversion (only what we need)
        converted = convert_video_if_needed(
            st.session_state.video_path,
            start_time=None,  # Convert full video for now
            duration=duration_sec_temp
        )
        if converted:
            st.session_state.converted_path = converted
            fps, total_frames = get_video_info(converted)
            st.session_state.fps = fps
            st.session_state.total_frames = total_frames

    # Video Range Selector
    if st.session_state.converted_path:
        st.subheader("Video Range Selector")
        fps = st.session_state.fps
        total_frames = st.session_state.total_frames
        duration_sec = total_frames / fps

        st.info(f"Video: {total_frames} frames @ {fps:.2f} FPS ({duration_sec:.1f} seconds)")

        col1, col2 = st.columns(2)
        with col1:
            start_time = st.slider("Start (seconds)", 0.0, duration_sec, 0.0, 0.1)
        with col2:
            end_time = st.slider("End (seconds)", 0.0, duration_sec, duration_sec, 0.1)

        # Clear ROI hash when start/end changes to force re-preview
        if 'last_roi_hash' in st.session_state:
            del st.session_state.last_roi_hash

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        st.write(f"Selected: Frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")

        # ROI Selector (Region of Interest)
        st.subheader("Region of Interest (ROI) - Focus Area")
        st.caption("Define the area where the person is performing the exercise")

        # Get video dimensions
        cap_info = cv2.VideoCapture(st.session_state.converted_path)
        frame_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            roi_x = st.number_input("ROI X", value=0, min_value=0, max_value=frame_w-100)
        with col2:
            roi_y = st.number_input("ROI Y", value=0, min_value=0, max_value=frame_h-100)
        with col3:
            roi_w = st.number_input("ROI Width", value=frame_w, min_value=100, max_value=frame_w)
        with col4:
            roi_h = st.number_input("ROI Height", value=frame_h, min_value=100, max_value=frame_h)

        # Store ROI
        st.session_state.roi = (roi_x, roi_y, roi_w, roi_h)


        # Video Player Preview with ROI overlay
        st.subheader("Video Preview (with ROI overlay)")

        # Auto-generate preview when ROI changes (limits to 5s)
        if st.session_state.roi:
            # Create a hash of ROI + start_time to detect changes
            roi_hash = hashlib.md5(f"{st.session_state.roi}-{start_time}".encode()).hexdigest()[:8]

            if 'last_roi_hash' not in st.session_state or st.session_state.last_roi_hash != roi_hash:
                st.session_state.last_roi_hash = roi_hash

                with st.spinner("Auto-generating ROI preview..."):
                    temp_dir = tempfile.mkdtemp()
                    preview_path = f"{temp_dir}/preview_roi.mp4"

                    x, y, w, h = st.session_state.roi

                    # Use selected range duration (not hardcoded 5s)
                    duration = end_time - start_time
                    start = start_time

                    subprocess.run([
                        "ffmpeg", "-ss", f"{start:.2f}", "-i", st.session_state.converted_path,
                        "-t", f"{duration:.2f}",
                        "-vf", f"drawbox=x={x}:y={y}:w={w}:h={h}:color=blue@0.8:t=5",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                        "-c:a", "aac",
                        "-y", preview_path
                    ], capture_output=True, timeout=30, check=False)

                    if os.path.exists(preview_path):
                        st.session_state.roi_preview_path = preview_path

            # Show the preview video
            if 'roi_preview_path' in st.session_state and os.path.exists(st.session_state.roi_preview_path):
                col_video = st.columns([1, 1, 1])[1]
                with col_video:
                    st.video(st.session_state.roi_preview_path)
            else:
                st.warning("ROI preview not generated yet")
        else:
            st.info("Set ROI values to see preview")

        # Skeleton Preview - now shows the ROI preview instead
        st.info("Use 'Video Preview (with ROI overlay)' above for skeleton reference")


        # Extract Button
        st.subheader("Extract Landmarks")
        if st.button("Extract Landmarks", type="primary"):
            progress_bar = st.progress(0)
            with st.spinner("Extracting landmarks with MediaPipe..."):
                result = extract_landmarks(
                    st.session_state.converted_path,
                    start_frame,
                    end_frame,
                    progress_bar,
                    st.session_state.roi if 'roi' in st.session_state else None
                )

                if result[0] is None:
                    st.error("Failed to extract landmarks")
                else:
                    landmarks, frames = result
                    st.session_state.landmarks = landmarks
                    st.session_state.frames = frames
                    st.session_state.start_frame = 0
                    st.session_state.end_frame = len(frames) - 1

                    with st.spinner("Normalizing..."):
                        normalized = normalize_skeleton(landmarks)
                        st.session_state.normalized = normalized

                    with st.spinner("Smoothing..."):
                        # Smooth original landmarks for display (0-1 range)
                        smoothed_display = smooth_landmarks(landmarks.copy())
                        st.session_state.smoothed_display = smoothed_display
                        # Smooth normalized for output
                        smoothed = smooth_landmarks(normalized)
                        st.session_state.smoothed = smoothed

                    with st.spinner("Detecting checkpoints..."):
                        checkpoints = detect_checkpoints(
                            smoothed,
                            0,
                            len(smoothed) - 1
                        )
                        st.session_state.checkpoints = checkpoints

                    st.success("Extraction complete!")

    # ================= POST-EXTRACTION UI =================
    if st.session_state.smoothed is not None:
        st.divider()
        st.header("Edit Extracted Data")

        # Rep Range Sliders
        n_frames = len(st.session_state.smoothed)
        st.subheader("Rep Range (Processed)")
        col1, col2 = st.columns(2)
        with col1:
            start = st.slider("Start Frame", 0, n_frames-1,
                            st.session_state.start_frame, key="start_slider")
        with col2:
            end = st.slider("End Frame", 0, n_frames-1,
                          st.session_state.end_frame, key="end_slider")

        st.session_state.start_frame = start
        st.session_state.end_frame = end

        if st.button("Update Checkpoints"):
            checkpoints = detect_checkpoints(
                st.session_state.smoothed, start, end
            )
            st.session_state.checkpoints = checkpoints

        # Checkpoints
        st.subheader("Checkpoints")
        if st.session_state.checkpoints:
            cp = st.session_state.checkpoints
            col1, col2, col3 = st.columns(3)
            with col1:
                cp['start'] = st.number_input("Start Frame",
                    value=cp['start'], min_value=start, max_value=end)
            with col2:
                cp['bottom'] = st.number_input("Bottom Frame",
                    value=cp['bottom'], min_value=start, max_value=end)
            with col3:
                cp['lockout'] = st.number_input("Lockout Frame",
                    value=cp['lockout'], min_value=start, max_value=end)

        # Export
        st.subheader("Export")
        if st.button("Export to JSON"):
            export_data = {
                'start_frame': int(st.session_state.checkpoints['start']),
                'bottom_frame': int(st.session_state.checkpoints['bottom']),
                'lockout_frame': int(st.session_state.checkpoints['lockout']),
                'normalized_landmarks': st.session_state.normalized[
                    st.session_state.start_frame:st.session_state.end_frame+1
                ].tolist(),
                'smoothed_landmarks': st.session_state.smoothed[
                    st.session_state.start_frame:st.session_state.end_frame+1
                ].tolist()
            }

            output_path = "reference_squat.json"
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            st.success(f"Exported to {output_path}")

        # Frame Viewer + Landmark Editor
        st.divider()
        col1, col2 = st.columns([1, 1])  # Both columns equal width (1/2 page each)

        with col1:
            st.subheader("Video + Skeleton")
            frame_idx = st.slider("Frame", 0, len(st.session_state.frames)-1, 0)

            if frame_idx < len(st.session_state.frames):
                frame = st.session_state.frames[frame_idx]
                if st.session_state.smoothed_display is not None and frame_idx < len(st.session_state.smoothed_display):
                    # Use ORIGINAL frame with SMOOTHED landmarks (0-1 range)
                    # Pass frame dimensions for proper coordinate scaling
                    annotated = draw_skeleton_on_frame(
                        frame, st.session_state.smoothed_display[frame_idx],
                        width=frame.shape[1], height=frame.shape[0]
                    )
                    # Show in a narrow column (1/3 of the column)
                    col_img = st.columns([1, 2, 1])[1]
                    with col_img:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               channels="RGB", width=640)
                elif st.session_state.landmarks is not None and frame_idx < len(st.session_state.landmarks):
                    # Fallback to original landmarks
                    annotated = draw_skeleton_on_frame(
                        frame, st.session_state.landmarks[frame_idx],
                        width=frame.shape[1], height=frame.shape[0]
                    )
                    col_img = st.columns([1, 2, 1])[1]
                    with col_img:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               channels="RGB", width=640)

        with col2:
            st.subheader("Landmark Editor")
            edit_frame = st.number_input("Edit Frame",
                0, len(st.session_state.smoothed)-1, frame_idx)

            if edit_frame < len(st.session_state.smoothed):
                lm_idx = st.selectbox("Landmark Index",
                    range(33), format_func=lambda x: f"{x}: {LANDMARK_NAMES.get(x, f'Landmark {x}')}")

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    x = st.number_input("X",
                        value=float(st.session_state.smoothed[edit_frame][lm_idx][0]))
                with col_b:
                    y = st.number_input("Y",
                        value=float(st.session_state.smoothed[edit_frame][lm_idx][1]))

                z = st.number_input("Z",
                    value=float(st.session_state.smoothed[edit_frame][lm_idx][2]))
                vis = st.number_input("Visibility", 0.0, 1.0,
                    value=float(st.session_state.smoothed[edit_frame][lm_idx][3]))

                if st.button("Update Landmark"):
                    st.session_state.smoothed[edit_frame][lm_idx] = [x, y, z, vis]
                    st.success("Landmark updated!")

        # Knee Angle Plot
        st.subheader("Knee Angle Over Time")
        angles = []
        frames_range = range(st.session_state.start_frame,
                           min(st.session_state.end_frame + 1,
                               len(st.session_state.smoothed)))
        for i in frames_range:
            angles.append(calculate_knee_angle(st.session_state.smoothed, i))

        if angles:
            st.line_chart(angles)

        # Skeleton Video Preview (like notebook)
        st.divider()
        st.subheader("Skeleton Video Preview")
        st.caption("Renders a video with only the skeleton overlay (no original video)")

        if st.button("Generate Skeleton Video"):
            if st.session_state.smoothed_display is not None:
                with st.spinner("Rendering skeleton video..."):
                    # Get video dimensions from first frame
                    if st.session_state.frames and len(st.session_state.frames) > 0:
                        h, w = st.session_state.frames[0].shape[:2]
                    else:
                        h, w = 640, 360

                    output_path = "skeleton_preview.mp4"
                    render_skeleton_video(
                        st.session_state.smoothed_display,
                        output_path,
                        width=w,
                        height=h,
                        fps=st.session_state.fps
                    )

                    if os.path.exists(output_path):
                        st.success("Skeleton video generated!")
                        col_vid = st.columns([1, 2, 1])[1]
                        with col_vid:
                            st.video(output_path)
                    else:
                        st.error("Failed to generate skeleton video")
            else:
                st.warning("Please extract landmarks first")

if __name__ == "__main__":
    main()
