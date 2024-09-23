import glob
from typing import Callable

import av


def get_video_duration(video_path: str) -> float:
    """
    return the duration of the video in seconds.
    """
    fh = av.open(video_path)
    if len(fh.streams.video) == 0:
        raise ValueError("No video stream found in the file.")
    if len(fh.streams.video) > 1:
        raise ValueError("More than one video stream found in the file.")
    video = fh.streams.video[0]
    return float(video.duration * video.time_base)


def iterate_all_samples(
    f: Callable,
    file_pattern: str,
    dataset_dir: str = "/fp/homes01/u01/ec-jinyueg/felles3/Research/Users/Alexander/Still Standing/1-raw",
    strict_count: bool = True,
    ):
    """
    Iterate over all the samples in the dataset and apply the input function f.
    if strict_count is True, it will raise an error if the number of files found is not 365.
    """

    files_list = sorted(glob.glob(f"{dataset_dir}{file_pattern}", recursive=True))
    if strict_count:
        assert len(files_list) == 365, f"Expected 365 files with pattern {file_pattern}, got {len(files_list)}."

    output = []
    for file in files_list:
        output.append(f(file))

    return output
