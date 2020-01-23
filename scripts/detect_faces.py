from dsfd import detect
from DFDlib.Utils.video import detect_faces_on_video

in_path = '/home/mc/dev/Deepfake-Detection-Challenge/data/train_sample_videos/aapnvogymq.mp4'
out_path = '/home/mc/Desktop/ann_aapnvogymq.mp4'

detector = detect.DSFDDetector(weight_path='/home/mc/dev/DSFD-Pytorch-Inference/dsfd/weights/WIDERFace_DSFD_RES152.pth')

detect_faces_on_video(detector, in_path, out_path, conf_thres=0.3)