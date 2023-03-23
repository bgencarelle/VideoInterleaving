import cv2
import numpy as np

def process_frames(frame_list):
    first_frame = cv2.imread(frame_list[0], cv2.IMREAD_UNCHANGED)
    height, width, _ = first_frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'avc1'), 30, (width, height), isColor=True)

    for index, png in enumerate(frame_list):
        print(f"Processing frame {index + 1}/{len(frame_list)}: {png}")

        frame = cv2.imread(png, cv2.IMREAD_UNCHANGED)

        bgra = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        bgr_bg = cv2.cvtColor(bgra, cv2.COLOR_RGBA2BGR)

        bgr_bg[bgra[:, :, 3] == 0] = [0, 255, 0]

        video.write(bgr_bg)

        for _ in range(black_frames_between):
            video.write(np.zeros((height, width, 3), dtype=np.uint8))

    video.release()
