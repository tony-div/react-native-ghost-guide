import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import streamlit as st


class CaptureState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.capture_active = False
        self.recording_active = False
        self.recording_path: Path | None = None
        self.input_format = 'v4l2'
        self.input_device = '/dev/video0'
        self.fps = 30
        self.size = '1280x720'
        self.preview_skip = 1
        self.frame = None
        self.writer: cv2.VideoWriter | None = None


CAPTURE_WORKERS: dict[str, CaptureState] = {}


def get_session_id() -> str:
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_capture_state() -> CaptureState:
    session_id = get_session_id()
    if session_id not in CAPTURE_WORKERS:
        CAPTURE_WORKERS[session_id] = CaptureState()
    return CAPTURE_WORKERS[session_id]


def capture_loop(state: CaptureState) -> None:
    cap = None
    try:
        while True:
            with state.lock:
                if not state.capture_active:
                    break
                input_format = state.input_format
                input_device = state.input_device
                fps = state.fps
                size = state.size
                preview_skip = state.preview_skip
                recording_active = state.recording_active
                recording_path = state.recording_path

            if cap is None:
                if input_format == 'v4l2':
                    cap = cv2.VideoCapture(input_device)
                elif input_format == 'avfoundation':
                    try:
                        cap = cv2.VideoCapture(int(input_device))
                    except ValueError:
                        cap = cv2.VideoCapture(0)
                elif input_format == 'dshow':
                    cap = cv2.VideoCapture(0)
                if cap is None or not cap.isOpened():
                    time.sleep(0.25)
                    continue
                width, height = size.split('x')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(int(preview_skip)):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with state.lock:
                state.frame = frame
                writer = state.writer
                if recording_active and writer is None and recording_path is not None:
                    width, height = size.split('x')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(
                        str(recording_path),
                        fourcc,
                        fps,
                        (int(width), int(height)),
                    )
                    state.writer = writer
                if not recording_active and writer is not None:
                    writer.release()
                    state.writer = None

            if recording_active and writer is not None:
                writer.write(frame)

            time.sleep(max(0.001, 1.0 / max(1, fps)))
    finally:
        with state.lock:
            if state.writer is not None:
                state.writer.release()
                state.writer = None
            state.running = False
            state.capture_active = False
        if cap is not None:
            cap.release()


def start_capture(state: CaptureState) -> None:
    with state.lock:
        if state.running:
            state.capture_active = True
            return
        state.capture_active = True
        state.running = True
    thread = threading.Thread(target=capture_loop, args=(state,), daemon=True)
    thread.start()


def request_rerun() -> bool:
    if hasattr(st, 'rerun'):
        st.rerun()
        return True
    if hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
        return True
    return False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_duration_seconds(video_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v',
                'error',
                '-show_entries',
                'format=duration',
                '-of',
                'default=noprint_wrappers=1:nokey=1',
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def record_video(
    output_path: Path,
    input_format: str,
    input_device: str,
    fps: int,
    size: str,
    duration: float,
) -> tuple[bool, str]:
    cmd = ['ffmpeg', '-y']

    if input_format == 'v4l2':
        cmd += [
            '-f',
            'v4l2',
            '-framerate',
            str(fps),
            '-video_size',
            size,
            '-i',
            input_device,
        ]
    elif input_format == 'avfoundation':
        cmd += [
            '-f',
            'avfoundation',
            '-framerate',
            str(fps),
            '-i',
            input_device,
        ]
    elif input_format == 'dshow':
        cmd += [
            '-f',
            'dshow',
            '-framerate',
            str(fps),
            '-video_size',
            size,
            '-i',
            input_device,
        ]
    else:
        return False, f'Unsupported input format: {input_format}'

    cmd += [
        '-t',
        f'{duration:.2f}',
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-crf',
        '23',
        '-pix_fmt',
        'yuv420p',
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or 'ffmpeg failed'
    return True, 'Recording complete'


def start_recording(
    output_path: Path,
    input_format: str,
    input_device: str,
    fps: int,
    size: str,
    duration: float,
) -> tuple[int | None, str]:
    cmd = ['ffmpeg', '-y']

    if input_format == 'v4l2':
        cmd += [
            '-f',
            'v4l2',
            '-framerate',
            str(fps),
            '-video_size',
            size,
            '-i',
            input_device,
        ]
    elif input_format == 'avfoundation':
        cmd += [
            '-f',
            'avfoundation',
            '-framerate',
            str(fps),
            '-i',
            input_device,
        ]
    elif input_format == 'dshow':
        cmd += [
            '-f',
            'dshow',
            '-framerate',
            str(fps),
            '-video_size',
            size,
            '-i',
            input_device,
        ]
    else:
        return None, f'Unsupported input format: {input_format}'

    if duration > 0:
        cmd += ['-t', f'{duration:.2f}']

    cmd += [
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-crf',
        '23',
        '-pix_fmt',
        'yuv420p',
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc.pid, 'Recording started'
    except Exception as exc:
        return None, f'Failed to start recording: {exc}'


def stop_recording(pid: int) -> tuple[bool, str]:
    try:
        os.kill(pid, signal.SIGTERM)
        return True, 'Recording stopped'
    except Exception as exc:
        return False, f'Failed to stop recording: {exc}'




def cut_video(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
) -> tuple[bool, str]:
    duration = max(0.0, end_time - start_time)
    cmd = [
        'ffmpeg',
        '-y',
        '-ss',
        f'{start_time:.2f}',
        '-i',
        str(input_path),
        '-t',
        f'{duration:.2f}',
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-crf',
        '23',
        '-c:a',
        'aac',
        '-movflags',
        '+faststart',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or 'ffmpeg failed'
    return True, 'Cut complete'


def main() -> None:
    st.set_page_config(page_title='Benchmark Rep Recorder', layout='wide')
    st.title('Benchmark Rep Recorder')

    st.info(
        'Record a benchmark rep video and cut a clean segment for reference.'
    )

    output_root = Path(
        st.text_input('Output directory', value='tools/recordings')
    )
    ensure_dir(output_root)

    st.header('Record')
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        exercise_name = st.text_input('Exercise name', value='bicep_curl')
        duration = st.number_input('Duration (seconds)', 1.0, 120.0, 10.0, 0.5)
        allow_manual_stop = st.checkbox('Allow manual stop', value=True)
        record_with_preview = st.checkbox('Record with preview', value=True)
    with col_b:
        fps = st.number_input('FPS', 1, 120, 30, 1)
        size = st.text_input('Resolution', value='1280x720')
    with col_c:
        input_format = st.selectbox(
            'Input format',
            ['v4l2', 'avfoundation', 'dshow'],
            index=0,
            help='v4l2=Linux, avfoundation=macOS, dshow=Windows',
        )
        input_device = st.text_input(
            'Input device',
            value='/dev/video0',
            help='avfoundation example: "0". dshow example: video="Integrated Camera"',
        )

    st.subheader('Camera Preview')
    preview_col1, preview_col2 = st.columns([1, 3])
    if 'preview_active' not in st.session_state:
        st.session_state.preview_active = False
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False

    if 'preview_started_at' not in st.session_state:
        st.session_state.preview_started_at = None

    with preview_col1:
        enable_preview = st.checkbox('Enable preview', value=True)
        preview_fps = st.number_input('Preview FPS', 1, 60, 15, 1)
        preview_width = st.number_input('Preview width', 160, 1920, 640, 10)
        preview_seconds = st.number_input('Preview seconds (0 = continuous)', 0, 120, 0, 1)
        preview_skip = st.number_input('Skip frames', 0, 10, 1, 1)
        start_preview = st.button('Start preview')
        stop_preview = st.button('Stop preview')
        if start_preview:
            st.session_state.preview_active = True
            st.session_state.preview_started_at = time.time()
        if stop_preview:
            st.session_state.preview_active = False
            st.session_state.preview_started_at = None
    with preview_col2:
        preview_placeholder = st.empty()

    if enable_preview and st.session_state.preview_active:
        state = get_capture_state()
        with state.lock:
            state.input_format = input_format
            state.input_device = input_device
            state.fps = int(preview_fps)
            state.size = size
            state.preview_skip = int(preview_skip)
        start_capture(state)

        if preview_seconds > 0 and st.session_state.preview_started_at is not None:
            if time.time() - st.session_state.preview_started_at >= float(preview_seconds):
                st.session_state.preview_active = False
                st.session_state.preview_started_at = None

        if st.session_state.preview_active:
            frame = None
            with state.lock:
                frame = None if state.frame is None else state.frame.copy()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(frame, width=int(preview_width))
            time.sleep(0.1)
            if not request_rerun():
                st.session_state.preview_active = False

    record_col1, record_col2 = st.columns([1, 3])
    with record_col1:
        if st.button('Start Recording', type='primary'):
            if st.session_state.get('recording_pid'):
                st.warning('Recording already in progress')
            else:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                output_path = output_root / f'{exercise_name}_{timestamp}.mp4'
                if record_with_preview:
                    state = get_capture_state()
                    with state.lock:
                        state.input_format = input_format
                        state.input_device = input_device
                        state.fps = int(fps)
                        state.size = size
                        state.recording_active = True
                        state.recording_path = output_path
                    start_capture(state)
                    st.session_state.recording_mode = 'shared'
                    st.session_state.recording_path = str(output_path)
                    st.session_state.recording_active = True
                    st.session_state.preview_active = True
                    st.success(f'Recording... {output_path.name}')
                else:
                    max_duration = 0 if allow_manual_stop else float(duration)
                    pid, message = start_recording(
                        output_path,
                        input_format,
                        input_device,
                        int(fps),
                        size,
                        max_duration,
                    )
                    if pid:
                        st.session_state.recording_mode = 'ffmpeg'
                        st.session_state.recording_pid = pid
                        st.session_state.recording_path = str(output_path)
                        st.success(f'Recording... {output_path.name}')
                    else:
                        st.error(message)

        if st.button('Stop Recording'):
            mode = st.session_state.get('recording_mode')
            if mode == 'shared':
                state = get_capture_state()
                with state.lock:
                    state.recording_active = False
                    state.recording_path = None
                saved_path = st.session_state.get('recording_path')
                if saved_path:
                    st.session_state.latest_video = saved_path
                    st.success(f'Saved: {saved_path}')
                st.session_state.recording_active = False
                st.session_state.recording_mode = None
                st.session_state.recording_path = None
            elif mode == 'ffmpeg':
                pid = st.session_state.get('recording_pid')
                if not pid:
                    st.warning('No active recording')
                else:
                    ok, message = stop_recording(pid)
                    if ok:
                        saved_path = st.session_state.get('recording_path')
                        if saved_path:
                            st.session_state.latest_video = saved_path
                            st.success(f'Saved: {saved_path}')
                        st.session_state.recording_pid = None
                        st.session_state.recording_path = None
                        st.session_state.recording_mode = None
                    else:
                        st.error(message)
            else:
                st.warning('No active recording')

    st.header('Cut')

    uploaded = st.file_uploader('Upload a video to cut', type=['mp4'])
    if uploaded:
        temp_path = output_root / f'upload_{uploaded.name}'
        with open(temp_path, 'wb') as f:
            f.write(uploaded.read())
        selected_path = temp_path
    else:
        recordings = sorted(output_root.glob('*.mp4'))
        selected_path = None
        if recordings:
            default_index = 0
            if 'latest_video' in st.session_state:
                latest = Path(st.session_state.latest_video)
                if latest in recordings:
                    default_index = recordings.index(latest)
            chosen = st.selectbox(
                'Pick a recording',
                recordings,
                index=default_index,
                format_func=lambda p: p.name,
            )
            selected_path = chosen

    if selected_path:
        st.video(str(selected_path))
        detected_duration = get_duration_seconds(selected_path)
        if detected_duration is None:
            detected_duration = st.number_input(
                'Video duration (seconds)', 0.1, 600.0, 10.0, 0.1
            )

        start_time, end_time = st.slider(
            'Cut range (seconds)',
            0.0,
            float(detected_duration),
            (0.0, float(detected_duration)),
            0.1,
        )

        cut_name = st.text_input('Cut name', value='rep_cut')
        cut_output = output_root / f'{cut_name}.mp4'

        if st.button('Cut Video', type='primary'):
            with st.spinner('Cutting...'):
                ok, message = cut_video(
                    selected_path,
                    cut_output,
                    float(start_time),
                    float(end_time),
                )
            if ok:
                st.success(f'Saved: {cut_output}')
                st.video(str(cut_output))
            else:
                st.error(message)
    else:
        st.info('Record or upload a video to enable cutting.')


if __name__ == '__main__':
    main()
