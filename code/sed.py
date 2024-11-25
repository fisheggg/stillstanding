import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import logging
from panns_inference import SoundEventDetection, labels
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from tqdm import tqdm

# dataset_path = "/home/arthur/felles/Research/Users/Alexander/Still Standing"
dataset_path = "/fp/homes01/u01/ec-jinyueg/felles2/Research/Users/Alexander/Still Standing"
metadata_path = "../metadata_processed.csv"
metadata = pd.read_csv(metadata_path)


def plot_sound_event_detection_result(
    stillstanding_no, framewise_output, out_fig_path=None, sr=32000, figsize=(12, 4), top_n=5
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


if __name__ == "__main__":
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
            audio_path = dataset_path + "/" + metadata.iloc[i-1]["audio_processed_path"]
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
