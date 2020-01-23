from tqdm import trange
import pandas as pd
from DFDlib.Utils.video import get_video_info
from DFDlib.Utils.logging import create_log, write_log

# Create log
log_path = create_log('get_video_info')

# Load master dataframe
in_df_path = '/home/mc/dev/Deepfake-Detection-Challenge/notebooks/master_dataframe.pkl'
out_df_path = '/home/mc/dev/Deepfake-Detection-Challenge/notebooks/master_dataframe_updated.pkl'
df = pd.read_pickle(in_df_path)
write_log(log_path, 'Master dataframe loaded')

for i in trange(len(df)):
    in_path = df.loc[i, 'filepath']

    # Get video info
    try:
        width, height, fps, num_frames = get_video_info(path=in_path)
    except:
        write_log(log_path, f'Error sampling from {in_path}')
        continue

    # Write new values to dataframe
    try:
        df.at[i, 'width'] = width
        df.at[i, 'height'] = height
        df.at[i, 'fps'] = fps
        df.at[i, 'num_frames'] = num_frames
        df.to_pickle(out_df_path, protocol=-1)
    except:
        write_log(log_path, f'Error writing to dataframe for {in_path}')
        continue

write_log(log_path, 'Finished')