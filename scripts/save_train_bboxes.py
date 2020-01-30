# Save bounding boxes for detected faces in training set

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from dsfd import detect
from DFDlib.Utils.video import get_face_bboxes
from DFDlib.Utils.logging import create_log, write_log

df = pd.read_pickle('/home/mc/dev/Deepfake-Detection-Challenge/notebooks/master_dataframe_updated.pkl')
detector = detect.DSFDDetector(weight_path='/home/mc/dev/DSFD-Pytorch-Inference/dsfd/weights/WIDERFace_DSFD_RES152.pth')

files = df.loc[:, 'filepath']
out_dir = '/media/mc/2TBNVMESSD/train_bboxes/'

# Create log
log_path = create_log('save_face_bboxes')

# Loop over file paths
for f in tqdm(files):
    # Create out path and check if it already exists
    out_path = (Path(out_dir) / Path(f).name).with_suffix('.npy')
    if out_path.exists(): continue
    
    # Get face bounding boxes
    try:
        bboxes = get_face_bboxes(detector, f, max_dim=512)
    except:
        msg = f'Error getting boxes for {f}'
        print(msg)
        write_log(log_path, msg)
        continue
    
    # Save bounding boxes
    try:
        np.save(out_path, bboxes)
    except:
        msg = f'Error writing bboxes as .npy for {f}'
        print(msg)
        write_log(log_path, msg)
        continue

write_log(log_path, 'Finished')