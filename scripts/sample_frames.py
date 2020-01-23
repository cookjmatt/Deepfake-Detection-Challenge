from tqdm import trange
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from DFDlib.Utils.video import sample_from_mp4
from DFDlib.Utils.logging import create_log, write_log

# Create log
log_path = create_log('sample_frames')

# Load master dataframe
in_df_path = '/home/mc/dev/Deepfake-Detection-Challenge/notebooks/master_dataframe.pkl'
out_df_path = '/home/mc/dev/Deepfake-Detection-Challenge/notebooks/master_dataframe_updated.pkl'
df = pd.read_pickle(in_df_path)
write_log(log_path, 'Master dataframe loaded')

# Define directory for saving sampled frames
save_dir = '/media/mc/2TBNVMESSD/sampled_frames/'

#for i in trange(len(df)):
for i in trange(50):
    in_path = df.loc[i, 'filepath']
    out_path = Path(save_dir) / Path(in_path).with_suffix('.npz').name

    # Skip is already sampled
    if out_path.exists(): continue

    # Sample frames mp4
    try:
        width, height, fps, num_frames, idxs, imgs = sample_from_mp4(path=in_path, samples=3, resize=None)
    except:
        write_log(log_path, f'Error sampling from {in_path}')
        continue

    # Write new values to dataframe
    try:
        df.at[i, 'orig_width'] = width
        df.at[i, 'orig_height'] = height
        df.at[i, 'new_width'] = imgs.shape[2]
        df.at[i, 'new_height'] = imgs.shape[1]
        df.at[i, 'fps'] = fps
        df.at[i, 'num_frames'] = num_frames
        df.at[i, 'idxs'] = np.array_str(idxs)
        df.to_pickle(out_df_path, protocol=-1)
    except:
        write_log(log_path, f'Error writing to dataframe for {in_path}')
        continue

    # Save file as npz
    try:
        np.savez(out_path, img=imgs)
    except:
        write_log(log_path, f'Error writing npz for {in_path}')
        continue

write_log(log_path, 'Finished')