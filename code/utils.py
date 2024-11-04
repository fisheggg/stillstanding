import re
import glob
from typing import Callable
from datetime import datetime, timedelta

import av
import pandas as pd


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


def clean_watch_data(
    input_csv_path: str, output_csv_path: str, start_offset: int = 5, duration: int = 500
):
    """
    clean csv data from the Polar Vantage V sports watch.
    """
    watch_timecode_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    watch_timecode_match = re.search(watch_timecode_pattern, input_csv_path)
    watch_timecode_str = watch_timecode_match.group()
    watch_timecode_format = (
        "%Y-%m-%d_%H-%M-%S"  # This format should match the pattern in the filename
    )
    watch_timecode_datetime = datetime.strptime(
        watch_timecode_str, watch_timecode_format
    )
    watch_timecode = pd.Timestamp(watch_timecode_datetime)
    watch_data = pd.read_csv(
        input_csv_path, delimiter=",", low_memory=False, skiprows=2, usecols=[1, 2, 9]
    )

    # Convert the elapsed time in seconds to a timedelta
    watch_data["time_elapsed"] = pd.TimedeltaIndex(watch_data["Time"])
    # Add the timedelta to the start date to get the actual datetime
    watch_data["time_column"] = watch_timecode + watch_data["time_elapsed"]

    # trim the start of the data
    watch_start_trimmed = watch_data['time_column'].min() + pd.Timedelta(seconds=start_offset)
    watch_data = watch_data[watch_data['time_column'] > watch_start_trimmed]
    # trim the duration of the data
    watch_time_duration_trimmed = watch_data['time_column'].min() + pd.Timedelta(seconds=duration)
    # Keep only the rows where the time is less than or equal to that
    watch_data = watch_data[watch_data['time_column'] <= watch_time_duration_trimmed]

    # Determine the reference time (first timestamp in the column)
    watch_reference_time = watch_data['time_column'].iloc[0]
    # Subtract the reference time from the entire column to get a timedelta
    watch_time_difference = watch_data['time_column'] - watch_reference_time
    # Convert the timedelta to seconds
    watch_data['Time'] = watch_time_difference.dt.total_seconds()

    # save output csv file
    watch_data.to_csv(output_csv_path)
    print(f"=> Done! cleaned csv file saved to {output_csv_path}")

def clean_phone_data(input_csv_path: str):
    """
    clean csv data from Physics Toolbox Sensor Suite.
    """
    mobile_data = pd.read_csv(input_csv_path, delimiter=';',decimal=',', low_memory=False)
    mobile_timecode_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}.\d{2}.\d{2}'
    mobile_timecode_match = re.search(mobile_timecode_pattern, input_csv_path)
    mobile_timecode_str = mobile_timecode_match.group()
    mobile_timecode_format = "%Y-%m-%dT%H.%M.%S" # This format should match the pattern in the filename
    mobile_timecode_datetime = datetime.strptime(mobile_timecode_str, mobile_timecode_format)
    mobile_timecode = pd.Timestamp(mobile_timecode_datetime)
    mobile_data.drop(mobile_data.columns[-2:],axis=1,inplace=True)

    # Replacing n-dash with hyphen
    mobile_data.replace(to_replace='−', value='-',regex=True, inplace=True)
    # Replacing comma with dot
    mobile_data.replace(to_replace=',', value='.',regex=True, inplace=True)
    # Replacing NAN with zeros
    mobile_data.replace(to_replace='∞', value='0',regex=True, inplace=True)
    # Now that the data should have been formatted correctly, we can change to float64
    mobile_data=mobile_data.astype(float)

    # remove redundant rows
    mobile_data.drop_duplicates(subset=['ax','ay','az'], keep='first', inplace=True)


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
