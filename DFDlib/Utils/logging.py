import datetime

def create_log(msg):
    start_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    log_path = f'/home/mc/dev/Deepfake-Detection-Challenge/logs/{start_time}-{msg}.txt'
    with open(log_path, 'w') as f:
        f.write(f'{msg} log created at {start_time}\n')
    return log_path

def write_log(path, msg):
    curr_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    with open(path, 'a') as f:
        f.write(f'{curr_time} {msg}\n')