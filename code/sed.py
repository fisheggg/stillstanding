import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from panns_inference import SoundEventDetection, labels
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from tqdm import tqdm
from scipy.spatial import ConvexHull

sys.path.append(Path(__file__).resolve().parent)
from ambisonics.distance import (
    SphericalAmbisonicsVisualizer,
    SphericalAmbisonicsDecoder,
)
from ambisonics.position import Position

dataset_path = "/home/arthur/felles1/Research/Users/Alexander/Still Standing"
# dataset_path = "/fp/homes01/u01/ec-jinyueg/felles2/Research/Users/Alexander/Still Standing"
# metadata_path = "../metadata_processed.csv"
metadata_path = Path(__file__).resolve().parent.parent / "metadata_processed.csv"
metadata = pd.read_csv(metadata_path)


def plot_sound_event_detection_result(
    stillstanding_no,
    framewise_output,
    out_fig_path=None,
    sr=32000,
    figsize=(12, 4),
    top_n=5,
):
    """Visualization of sound event detection result.

    Args:
      framewise_output: (time_steps, classes_num)
    """
    if out_fig_path is not None:
        os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    classwise_output = np.max(framewise_output, axis=0)  # (classes_num,)

    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:top_n]

    ix_to_lb = {i: label for i, label in enumerate(labels)}
    time = librosa.times_like(framewise_output[:, 0], sr=32000, hop_length=320)
    lines = []
    fig, ax = plt.subplots(figsize=figsize)
    for idx in idxes:
        (line,) = ax.plot(time, framewise_output[:, idx], label=ix_to_lb[idx])
        lines.append(line)

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    plt.legend(handles=lines)
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title(f"Still Standing {stillstanding_no:03}")
    plt.ylim(0, 1.0)
    plt.grid(which="major", color="grey", linestyle="--", axis="x")
    plt.grid(which="minor", color="grey", linestyle=":", axis="x")
    if out_fig_path is not None:
        plt.savefig(out_fig_path)
        print("Save fig to {}".format(out_fig_path))


def run_sed():
    """Run sound event detection on the Still Standing dataset."""
    device = "cpu"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("../sed_results/inference.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Starting SED inference for stillstanding. Device: {device}")
    sed = SoundEventDetection(checkpoint_path=None, device=device)
    for i in range(249, 366):
        start_time = time.time()
        logging.info(f"Processing No.{i:03}")

        try:
            audio_path = (
                dataset_path + "/" + metadata.iloc[i - 1]["audio_processed_path"]
            )
        except Exception as e:
            logging.error(f"No audio for No.{i:03}: {e}")
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=32000, mono=False, duration=None)
        except Exception as e:
            logging.error(f"Failed to load audio for No.{i:03}: {e}")
            continue

        try:
            output = sed.inference(np.expand_dims(audio[0], axis=0))
        except Exception as e:
            logging.error(f"Failed during inferencing No.{i:03}: {e}")
            continue

        np.save(f"../sed_results/output/{i:03}.npy", output[0].astype(np.float16))

        logging.info(f"Processed No.{i:03} in {time.time() - start_time:.2f}s")


def compute_directional_audio(stillstanding_no, output_dir):
    """
    Compute the directional audio from the sound event detection output.
    First compute the directional distribution and find peak angles. Result are saved as a figure to output_dir.
    Then, use the computed peak angles to decode the ambisonics and generate directional output.
    Multiple audio files are saved to output_dir with filenames indicating the angle and summed relative energy.
    """
    # load original FoA audio
    logging.info(f"Loading audio for No.{stillstanding_no:03}")
    wave_path = os.path.join(
        dataset_path, metadata.iloc[stillstanding_no - 1].audio_processed_path
    )
    ambi, sr = librosa.load(wave_path, sr=None, mono=False, duration=None)

    # compute energy of each DoA
    logging.info(f"Computing DoA for No.{stillstanding_no:03}")
    viz = SphericalAmbisonicsVisualizer(
        data=ambi.T, rate=sr, angular_res=None, phi_res=2.0, nu_res=90
    )
    out_sum = []
    for out in viz.loop_frames():
        out_sum.append(out)
    out = np.zeros((len(out_sum), 180))
    for i, o in enumerate(out_sum):
        out[i] = o[1, :]
    out_sum = np.array(out_sum)

    # find peak angles
    logging.info(f"Finding peaks for No.{stillstanding_no:03}")
    angles = viz.mesh()[1][1, :]
    count_max = np.zeros_like(angles)
    count_max_energy = np.zeros_like(angles)
    argmaxes = out_sum[:, 1, :].argmax(axis=1)
    for frame, direction_idx in enumerate(argmaxes):
        count_max[direction_idx] += 1
        count_max_energy[direction_idx] += out_sum[frame, 1, direction_idx]
    hull_count = ConvexHull(
        np.vstack([count_max * np.cos(angles), count_max * np.sin(angles)]).T
    )
    hull_energy = ConvexHull(
        np.vstack(
            [count_max_energy * np.cos(angles), count_max_energy * np.sin(angles)]
        ).T
    )
    peak_count = librosa.util.peak_pick(
        count_max,
        pre_max=11,
        post_max=11,
        pre_avg=11,
        post_avg=11,
        delta=0.1 * count_max.max().item(),
        wait=11,
    )
    peak_energy = librosa.util.peak_pick(
        count_max_energy,
        pre_max=11,
        post_max=11,
        pre_avg=11,
        post_avg=11,
        delta=0.1 * count_max_energy.max().item(),
        wait=11,
    )

    # generate figure
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Energy direction of Still Standing {stillstanding_no:03}")
    ax1 = plt.subplot(1, 2, 1, projection="polar")
    ax1.set_theta_zero_location("N")
    ax1.bar(angles, count_max, label="Count of frames being max energy", width=0.02)
    ax1.plot(
        angles[hull_count.vertices],
        count_max[hull_count.vertices],
        label="Convex hull of count",
        alpha=0.9,
        linestyle="--",
        color="orange",
    )
    ax1.vlines(
        angles[peak_count],
        0,
        count_max.max(),
        color="red",
        linestyle=(0, (1, 1)),
        label="Peaks of count",
    )
    ax1.legend()
    ax2 = plt.subplot(1, 2, 2, projection="polar")
    ax2.set_theta_zero_location("N")
    ax2.bar(
        angles,
        count_max_energy,
        label="Sum of energy when being max direction",
        width=0.02,
    )
    ax2.plot(
        angles[hull_energy.vertices],
        count_max_energy[hull_energy.vertices],
        label="Convex hull of energy",
        alpha=1,
        linestyle="--",
        color="orange",
    )
    ax2.vlines(
        angles[peak_energy],
        0,
        count_max_energy.max(),
        color="red",
        linestyle=(0, (1, 1)),
        label="Peaks of energy",
        alpha=0.9,
    )
    ax2.legend()
    save_path = os.path.join(output_dir, f"{stillstanding_no:03}_doa.png")
    plt.savefig(fname=save_path, dpi=300)
    logging.info(f"Saved DoA figure for No.{stillstanding_no:03} to {save_path}")

    # decode ambisonics
    logging.info(f"Decoding ambisonics for No.{stillstanding_no:03}")
    position_list = [
        Position(phi / np.pi * 180.0, 0.0, 1.0, "polar") for phi in angles[peak_energy]
    ]
    ambi_dec = SphericalAmbisonicsDecoder(data=ambi.T, rate=sr, pos_list=position_list)
    output = ambi_dec.decode()
    num_files = output.shape[1]
    for i in range(num_files):
        save_path = os.path.join(
            output_dir,
            f"{stillstanding_no:03}_{i:03}_{angles[peak_energy][i]/np.pi*180:.0f}_{count_max_energy[peak_energy][i]:.2f}.wav",
        )
        sf.write(save_path, output[:, i], sr)

    logging.info(
        f"Saved {num_files} audio files for No.{stillstanding_no:03} to {output_dir}"
    )


def run_directional_audio(no_list=None):
    os.makedirs(os.path.join(dataset_path, "2-processed/doa_results"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(dataset_path, "2-processed/doa_results/compute.log")
            ),
        ],
    )
    if no_list is None:
        logging.info("Starting DoA inference for all stillstanding samples")
    else:
        logging.info(f"Starting DoA inference for given list: {no_list}")

    if no_list is None:
        no_list = range(1, 366)

    for i in no_list:
        logging.info(f"Processing No.{i:03}")
        output_dir = os.path.join(dataset_path, f"2-processed/doa_results/{i:03}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            compute_directional_audio(i, output_dir)
        except Exception as e:
            logging.error(f"Error during computing {i:03}: {e}")


if __name__ == "__main__":
    # run_sed()
    run_directional_audio(range(20, 365))
    # run_directional_audio()
