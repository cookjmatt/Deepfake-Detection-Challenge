import numpy as np
from tqdm import trange
import cv2
from dsfd import detect
from DFDlib.Utils.image import resize_to_max

# Get video info
def get_video_info(path):
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, num_frames

# Sample evenly distributed frames from an mp4 file and resize to a max dimension
def sample_from_mp4(path, samples, resize=None):
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, num_frames-1, samples, dtype=int)
    frames = []
    for i in range(0, num_frames):
        ret = cap.grab()
        if i in idxs:
            ret, frame = cap.retrieve()
            if resize is not None:
                frame = resize_to_max(frame, resize)
            frames.append(frame)
    cap.release()
    return width, height, fps, num_frames, idxs, np.stack(frames)

# Draw faces on an image
def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def detect_faces_on_video(detector, in_path, out_path, conf_thres=0.3, shrink=0.5):
    # Setup in_path capture and get params
    cap = cv2.VideoCapture(in_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup out_path writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Loop over capture frames
    for i in trange(num_frames):
        # Get next frame and break if empty
        ret, frame = cap.read()
        if not ret: break

        # Detect faces and draw bounding box on frame
        dets = detector.detect_face(frame, conf_thres, shrink)[:, :4]
        draw_faces(frame, dets)

        # Write out frame
        out.write(frame)
    
    # Release videos
    cap.release()
    out.release()

# Convert a video to numpy
def mp4_to_npz(in_path):
    cap = cv2.VideoCapture(in_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if frame is None: break
        frames.append(frame)
    return np.stack(frames)

# Get face bboxes for a video
def get_face_bboxes(detector, in_path, conf_thres=0.3, max_dim=None):
    # Get video capture and parameters
    cap = cv2.VideoCapture(in_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate shrink factor from max image dimension and max wanted dimension
    shrink = 1.0
    if max_dim is not None:
        max_img = max(width, height)
        if max_img > max_dim:
            shrink = max_dim / max_img

    # Get face bounding boxes
    bboxes = []
    for i in range(num_frames):
        ret, frame = cap.read()
        dets = detector.detect_face(frame, conf_thres, shrink)
        bboxes.append(dets)
    return bboxes