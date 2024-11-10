import os
import re
import glob
from typing import Callable
from datetime import datetime, timedelta

import av
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_video_duration(video_path: str) -> float:
    """
    return the duration of the video in seconds.
    """
    try:
        fh = av.open(video_path)
        if len(fh.streams.video) == 0:
            raise ValueError("No video stream found in the file.")
        # if len(fh.streams.video) > 1:
            # raise ValueError("More than one video stream found in the file.")
        video = fh.streams.video[0]
        return float(video.duration * video.time_base)
    except Exception as e:
        print(f"Error: {e}")
        return None

def downsample(df: pd.DataFrame, time_col, target_col, target_sr) -> pd.DataFrame:
    """
    Group and average entries within the target sampling rate interval.
    Assuming time_col starts at 0 and is in seconds.
    df: input dataframe
    time_col: column name of the time column
    target_col: column name of the target column
    target_sr: target sampling rate
    """
    interval = 1 / target_sr
    duration = (df[time_col].max() // interval) * interval + interval

    # Create a new time column with the target sampling rate
    time_slots = [float(i) for i in range(0, int(duration), int(interval))]
    value_lists = [[] for _ in range(len(time_slots))]
    for i, row in df.iterrows():
        time = row[time_col]
        idx = int(time // interval)
        value_lists[idx].append(row[target_col])

    # Create a new dataframe with the downsampled values
    downsampled_df = pd.DataFrame(
        {
            time_col: time_slots,
            target_col: [np.mean(values) if len(values) > 0 else 0 for values in value_lists],
        }
    )
    return downsampled_df


def plot_loudness_hr_brightness(stillstanding_no: int, dataset_dir: str, metadata_path: str = "../metadata.csv", save_path: str = None):
    """
    Plot the loudness and heart rate data for a given sample.
    """
    metadata = pd.read_csv(metadata_path)

    row = metadata[metadata.StillStandingNo == stillstanding_no].iloc[0]
    phone_path = os.path.join(dataset_dir, row.processed_phone_path)
    watch_path = os.path.join(dataset_dir, row.processed_watch_path)
    phone_data = pd.read_csv(phone_path)
    loudness = phone_data[['time', 'Gain']]
    loudness = downsample(loudness, 'time', 'Gain', 1)
    brightness = phone_data[['time', 'I']]
    brightness = downsample(brightness, 'time', 'I', 1)
    hr = pd.read_csv(watch_path)[['Time', 'HR (bpm)']].ffill()


    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    plt.plot(hr['Time'], hr['HR (bpm)'], label='HR')
    plt.title(f"Still Standing No. {stillstanding_no}, {row.Date}")
    plt.ylabel('Loudness (dB)')
    plt.grid()
    plt.subplot(312)
    plt.plot(brightness['time'],brightness['I'], label='Brightness')
    plt.ylabel('Brightness (lux)')
    plt.grid()
    plt.subplot(313)
    plt.plot(loudness['time'], loudness['Gain'], label='Loudness')
    plt.ylabel('Loudness (dB)')
    plt.xlabel('Time (s)')
    plt.grid()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        print(f"=> plot saved to {save_path}")



def iterate_all_samples(
    f: Callable,
    file_pattern: str,
    dataset_dir: str = "/home/arthur/felles/Research/Users/Alexander/Still Standing/1-raw",
    strict_count: bool = True,
):
    """
    Iterate over all the samples in the dataset and apply the input function f.
    if strict_count is True, it will raise an error if the number of files found is not 365.
    """

    files_list = sorted(glob.glob(f"{dataset_dir}{file_pattern}", recursive=True))
    if strict_count:
        assert (
            len(files_list) == 365
        ), f"Expected 365 files with pattern {file_pattern}, got {len(files_list)}."

    output = []
    for file in files_list:
        output.append(f(file))

    return output
