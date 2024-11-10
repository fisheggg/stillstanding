"""
Batch processed raw data
modified from original notebook
"""

import os
import re
import subprocess
from datetime import datetime, timedelta

import av
import maad
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import musicalgestures as mg
from scipy.signal import savgol_filter
from matplotlib.ticker import FormatStrFormatter


def process_stillstanding(metadata_path: str, output_dir: str, log_path: None):
    """
    Process all stillstanding data.
    """
    pass


def process_stillstanding_day(stillstanding_no: int, output_path: str, metadata_path: str = '../metadata_raw.csv'):
    """
    Process one day of stillstanding data, same function to the original notebook.
    """
    pass


def process_watch_data(
    stillstanding_no: int,
    input_csv_path: str,
    output_dir: str,
    start_offset: int = 5,
    duration: int = 500,
) -> datetime.strptime:
    """
    clean csv data from the Polar Vantage V sports watch.
    Processed csv is saved to output_csv_path.
    Returns the watch timecode of start time as a datetime object.
    """
    # find the timecode from the filename
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

    # load file
    watch_data = pd.read_csv(
        input_csv_path, delimiter=",", low_memory=False, skiprows=2, usecols=[1, 2, 9]
    )
    # Convert the elapsed time in seconds to a timedelta
    watch_data["time_elapsed"] = pd.TimedeltaIndex(watch_data["Time"])
    # Add the timedelta to the start date to get the actual datetime
    watch_data["time_column"] = watch_timecode + watch_data["time_elapsed"]

    # trim the start of the data
    watch_start_trimmed = watch_data["time_column"].min() + pd.Timedelta(
        seconds=start_offset
    )
    watch_data = watch_data[watch_data["time_column"] > watch_start_trimmed]
    # trim the duration of the data
    watch_time_duration_trimmed = watch_data["time_column"].min() + pd.Timedelta(
        seconds=duration
    )
    # Keep only the rows where the time is less than or equal to that
    watch_data = watch_data[watch_data["time_column"] <= watch_time_duration_trimmed]

    # Determine the reference time (first timestamp in the column)
    watch_reference_time = watch_data["time_column"].iloc[0]
    # Subtract the reference time from the entire column to get a timedelta
    watch_time_difference = watch_data["time_column"] - watch_reference_time
    # Convert the timedelta to seconds
    watch_data["Time"] = watch_time_difference.dt.total_seconds()

    # save output csv file
    output_csv_path = os.path.join(
        output_dir, f"stillstanding_{stillstanding_no}_watch.csv"
    )
    watch_data.to_csv(output_csv_path)
    print(f"=> Cleaned watch data saved to {output_csv_path}")

    return watch_timecode


def process_phone_data(
    stillstanding_no: int,
    input_csv_path: str,
    output_dir: str,
    clap_start_s: int,
    clap_end_s: int,
    duration: int = 500,
) -> datetime.strptime:
    """
    clean csv data from Physics Toolbox Sensor Suite.
    processed csv is saved to output_csv_path.
    Returns:
        phone_timecode: datetime object of the start time of the data.
        phone_sr: sampling rate of phone data.
    """
    # Find the timecode from the filename
    mobile_data = pd.read_csv(
        input_csv_path, delimiter=";", decimal=",", low_memory=False
    )
    mobile_timecode_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}.\d{2}.\d{2}"
    mobile_timecode_match = re.search(mobile_timecode_pattern, input_csv_path)
    mobile_timecode_str = mobile_timecode_match.group()
    mobile_timecode_format = (
        "%Y-%m-%dT%H.%M.%S"  # This format should match the pattern in the filename
    )
    mobile_timecode_datetime = datetime.strptime(
        mobile_timecode_str, mobile_timecode_format
    )
    mobile_timecode = pd.Timestamp(mobile_timecode_datetime)

    # read phone data
    mobile_data.drop(mobile_data.columns[-2:], axis=1, inplace=True)
    # Replacing n-dash with hyphen
    mobile_data.replace(to_replace="−", value="-", regex=True, inplace=True)
    # Replacing comma with dot
    mobile_data.replace(to_replace=",", value=".", regex=True, inplace=True)
    # Replacing NAN with zeros
    mobile_data.replace(to_replace="∞", value="0", regex=True, inplace=True)
    # Now that the data should have been formatted correctly, we can change to float64
    mobile_data = mobile_data.astype(float)
    # The sampling rate is the total number of samples divided by time
    mobile_sr = len(mobile_data) / mobile_data["time"].iloc[-1]
    print(f"=> Original sample rate of phone data: {mobile_sr:.2f}")
    # remove redundant rows
    mobile_data.drop_duplicates(subset=["ax", "ay", "az"], keep="first", inplace=True)
    mobile_sr_unique = len(mobile_data) / mobile_data["time"].iloc[-1]
    print(f"=> Unique sample rate of phone data: {mobile_sr_unique:.2f}")
    # Convert the elapsed time in seconds to a timedelta
    mobile_data["time_elapsed"] = pd.to_timedelta(mobile_data["time"], unit="s")
    # Add the timedelta to the start date to get the actual datetime
    mobile_data["time_column"] = mobile_timecode + mobile_data["time_elapsed"]

    mobile_clap_start_s = clap_start_s
    mobile_clap_end_s = clap_end_s
    print(
        f"=> Triming using clap start and end times: {mobile_clap_start_s} - {mobile_clap_end_s}"
    )
    # First we trim to the first clap
    mobile_time_after_first_clap = mobile_data["time_column"].min() + pd.Timedelta(
        seconds=mobile_clap_start_s
    )
    mobile_data_clap_to_clap = mobile_data[
        mobile_data["time_column"] > mobile_time_after_first_clap
    ]
    # Then we remove the next 20 seconds of data
    mobile_time_20_seconds_after = mobile_data_clap_to_clap[
        "time_column"
    ].min() + pd.Timedelta(seconds=20)
    mobile_data_standstill = mobile_data_clap_to_clap[
        mobile_data_clap_to_clap["time_column"] > mobile_time_20_seconds_after
    ]
    # Identify the time 500 seconds after the start
    mobile_time_trimmed = mobile_data_standstill["time_column"].min() + pd.Timedelta(
        seconds=duration
    )
    # Keep only the rows where the time is less than or equal to that
    mobile_data_standstill = mobile_data_standstill[
        mobile_data_standstill["time_column"] <= mobile_time_trimmed
    ]
    # Find the timecode of the first row
    mobile_first_timecode = mobile_data_standstill["time"].iloc[0]
    # Subtract that value from the entire time column
    mobile_data_standstill["time"] = (
        mobile_data_standstill["time"] - mobile_first_timecode
    )

    # save output csv file
    output_csv_path = os.path.join(
        output_dir, f"stillstanding_{stillstanding_no}_phone.csv"
    )
    mobile_data_standstill.to_csv(output_csv_path)
    print(f"=> Cleaned phone data saved to {output_csv_path}")
    return mobile_timecode, mobile_sr


def process_audio_data(
    stillstanding_no: int,
    input_audio_path: str,
    output_dir: str,
    clap_start_s: int,
    clap_end_s: int,
):
    """
    Trim audio files and save to output_dir.
    Four audio files are saved:
    1. synced audio
    2. synced and trimmed audio
    3. clap at the beginning
    4. clap at the end
    """
    audio_clap_start = str(timedelta(seconds=clap_start_s))
    audio_clap_end = str(timedelta(seconds=clap_end_s))
    print(
        f"=> Triming audio using clap start and end times: {audio_clap_start} - {audio_clap_end}"
    )

    # output synced audio
    output_synced_audio_path = os.path.join(
        output_dir, f"stillstanding_{stillstanding_no}_ambisonics.wav"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_audio_path,
            "-ss",
            audio_clap_start,
            "-to",
            audio_clap_end,
            "-c",
            "copy",
            output_synced_audio_path,
        ]
    )

    # output trimmed audio
    # Convert time string to a datetime object
    audio_time_object = datetime.strptime(audio_clap_start, "%H:%M:%S")
    # Add 20 seconds to the beginning and a 500 second duration
    audio_new_time_start = audio_time_object + timedelta(seconds=20)
    audio_new_time_end = audio_new_time_start + timedelta(seconds=500)
    # Convert back to a string
    audio_standstill_start = audio_new_time_start.strftime("%H:%M:%S")
    audio_standstill_end = audio_new_time_end.strftime("%H:%M:%S")
    output_trimmed_audio_path = os.path.join(
        output_dir, f"stillstanding_{stillstanding_no}_ambisonics_trim.wav"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_audio_path,
            "-ss",
            audio_standstill_start,
            "-to",
            audio_standstill_end,
            "-c",
            "copy",
            output_trimmed_audio_path,
        ]
    )

    # output claps
    # Convert time string to a datetime object
    audio_time_object_start = datetime.strptime(audio_clap_start, "%H:%M:%S")
    audio_time_object_end = datetime.strptime(audio_clap_end, "%H:%M:%S")
    # Add 5 seconds to the beginning
    audio_first_clap_end = audio_time_object_start + timedelta(seconds=5)
    # Remove 5 seconds from the end
    audio_last_clap_begin = audio_time_object_end - timedelta(seconds=5)
    # Convert back to a string
    audio_first_clap_end_time = audio_first_clap_end.strftime("%H:%M:%S")
    audio_last_clap_begin_time = audio_last_clap_begin.strftime("%H:%M:%S")
    audio_first_clap_fn = "stillstanding_" + str(stillstanding_no) + "_clap_first.wav"
    audio_last_clap_fn = "stillstanding_" + str(stillstanding_no) + "_clap_last.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_audio_path,
            "-ss",
            audio_clap_start,
            "-to",
            audio_first_clap_end_time,
            "-c",
            "copy",
            os.path.join(output_dir, audio_first_clap_fn),
        ]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_audio_path,
            "-ss",
            audio_last_clap_begin_time,
            "-to",
            audio_clap_end,
            "-c",
            "copy",
            os.path.join(output_dir, audio_last_clap_fn),
        ]
    )

    print(f"=> All audio files saved to {output_dir}")


def process_fisheye_video_data(
    stillstanding_no: int,
    input_video_list: tuple[str],
    output_dir: str,
    clap_start_s: str,
    clap_end_s: str,
):
    """
    Process fisheye video data.
    5 video files are saved:
    1. concatenated spherical video
    2. synced spherical video
    3. synced and trimmed spherical video
    4. synced and trimmed person shot
    5. synced and trimmed room shot

    Parameters
    ----------
    input_video_list : tuple[str]
        List of input video files.
    output_dir : str
        Directory to save the output files.
    stillstanding_no : int
        Stillstanding number.
    clap_start_s : int
        Start time of the clap in seconds.
    clap_end_s : int
        End time of the clap in seconds.
    person_shotcrop : tuple[int]
        Crop parameters for the person shot. (height, width, xpos, ypos)
    room_shotcrop : tuple[int]
        Crop parameters for the room shot. (height, width, xpos, ypos)
    """
    print(f"=> Processing video data, found {len(input_video_list)} videos")
    # create txt file with the list of videos
    list_path = os.path.join(output_dir, "mylist.txt")
    with open(list_path, "w") as f:
        for video in input_video_list:
            f.write(f"file '{video}'\n")

    # output concatenated video
    spherical_path = os.path.join(output_dir, f"stillstanding_{stillstanding_no}_spherical.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            spherical_path
        ]
    )

    # output synced video
    video_clap_start = clap_start_s
    video_clap_end = clap_end_s
    synced_path = os.path.join(output_dir, f"stillstanding_{stillstanding_no}_spherical_clap.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            spherical_path,
            "-ss",
            str(video_clap_start),
            "-to",
            str(video_clap_end),
            "-c",
            "copy",
            synced_path
        ]
    )

    # output synced and trimmed video
    trimmed_path = os.path.join(output_dir, f"stillstanding_{stillstanding_no}_spherical_trim.mp4")
    # Convert time string to a datetime object
    video_time_object = datetime.strptime(video_clap_start, '%H:%M:%S')
    # Add 20 seconds to the beginning and a 500 second duration
    video_new_time_start = video_time_object + timedelta(seconds=20)
    video_new_time_end = video_new_time_start + timedelta(seconds=500)
    # Convert back to a string
    video_standstill_start = video_new_time_start.strftime('%H:%M:%S')
    video_standstill_end = video_new_time_end.strftime('%H:%M:%S')
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            spherical_path,
            "-ss",
            video_standstill_start,
            "-to",
            video_standstill_end,
            "-c",
            "copy",
            trimmed_path
        ]
    )

    os.remove(list_path)
    print(f"=> All fisheye video files saved to {output_dir}")


def processed_360_video_data(
    stillstanding_no: int,
    input_video_list: tuple[str],
    output_dir: str,
    clap_start_s: str,
    clap_end_s: str,
    person_shotcrop: tuple[int],
    room_shotcrop: tuple[int],
):
    """
    Process 360 video data.
    2 video files are saved:
    """
    # create txt file with the list of videos
    list_path = os.path.join(output_dir, "mylist.txt")
    with open(list_path, "w") as f:
        for video in input_video_list:
            f.write(f"file '{video}'\n")

    # I have had problems finding a format that supports all the content of the original .360 files. 
    video_out_fn_trim = 'stillstanding_' + str(stillstanding_no) + '_trim.mkv'
    # Instead we create four separate tracks to keep the valuable media:
    video_out_fn_track0 = 'stillstanding_' + str(stillstanding_no) + '_trim_track0.mkv'
    video_out_fn_track5 = 'stillstanding_' + str(stillstanding_no) + '_trim_track5.mkv'
    video_out_fn_track1 = 'stillstanding_' + str(stillstanding_no) + '_trim_track1.aac'
    video_out_fn_track6 = 'stillstanding_' + str(stillstanding_no) + '_trim_track6.wav'

    # extract each of the audio and video tracks as separate files
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            list_path,
            '-ss',
            clap_start_s,
            '-to',
            clap_end_s,
            '-map',
            '0:0',
            '-map',
            '0:6', 
            '-c',
            'copy',
            video_out_fn_track0,
            '-ss',
            clap_start_s,
            '-to',
            clap_end_s,
            '-map',
            '0:5',
            '-map',
            '0:6',
            '-c',
            'copy',
            video_out_fn_track5,
            '-ss',
            clap_start_s,
            '-to',
            clap_end_s,
            '-map',
            '0:1',
            '-c',
            'copy',
            video_out_fn_track1,
            '-ss',
            clap_start_s,
            '-to',
            clap_end_s,
            '-map',
            '0:6',
            '-c',
            'copy',
            video_out_fn_track6 
        ]
    )

    # output person shot
    person_shot_fn_trim = 'stillstanding_' + str(stillstanding_no) + '_arj_trim_crop.mp4'
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            video_out_fn_track5,
            '-vf',
            f'crop={person_shotcrop[0]}:{person_shotcrop[1]}:{person_shotcrop[2]}:{person_shotcrop[3]},transpose=2',
            person_shot_fn_trim
        ]
    )

    # output room shot
    room_shot_fn_trim = 'stillstanding_' + str(stillstanding_no) + '_room_trim_crop.mp4'
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            video_out_fn_track0,
            '-vf',
            f'crop={room_shotcrop[0]}:{room_shotcrop[1]}:{room_shotcrop[2]}:{room_shotcrop[3]}',
            '-b:v',
            '9M',
            room_shot_fn_trim
        ]
    )

    os.remove(list_path)
    print(f"=> All 360 video files saved to {output_dir}")