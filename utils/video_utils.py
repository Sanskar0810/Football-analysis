import os
import cv2

def read_video(video_path):
    print(f"🔍 Checking file at: {video_path}")

    if not os.path.exists(video_path):
        raise ValueError(f"❌ Error: File not found at {video_path}")
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"❌ Error: Unable to open video file {video_path}")

    print("✅ OpenCV successfully opened the video!")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("❌ Error: No frames to save!")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    out = cv2.VideoWriter(output_video_path, fourcc, 24, frame_size)

    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
    print(f"✅ Video saved at: {output_video_path}")