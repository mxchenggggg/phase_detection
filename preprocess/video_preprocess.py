# Necessary import
import os
import cv2
import pandas as pd
import re
import click
import logging
from rich.logging import RichHandler
from rich.progress import track

FORMAT = "%(message)s"
logging.basicConfig(
    level="DEBUG",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger(__name__)


@click.command()
@click.option("-r", "--data-root", help="Where the root of video is.")
@click.option("-o", "--output-root", help="Where the root of output is.")
@click.option("-w", "--width", help="Where the width of image is.")
@click.option("-h", "--height", help="Where the height of image is.")
def main(data_root: str, output_root: str, width: int, height: int):
    """_summary_

    Args:
        data_root (str): Data root path where to store the raw videos
        output_root (str): Output path where to store preprocrssed frames
        width (int): The width of the frame 
        height (int): The height of the frame 

    Raises:
        Exception: Data root does not exist
    """

    # Check if the data root exist
    if not os.path.exists(data_root):
        raise Exception("Data root does not exist")

    # Check if such directory exist or not
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # Check what videos we have in the output root
    existing_videos = [
        tmp for tmp in os.listdir(output_root) if bool(re.match("[V][0-9][0-9]+", tmp))
    ]
    # Fetch the videos that are needed to be preprocessed
    videos_to_process = [
        tmp
        for tmp in os.listdir(data_root)
        if bool(re.match("[V][0-9][0-9]+", tmp)) and tmp not in existing_videos
    ]
    log.debug(f"Need to process {videos_to_process}")
    # Create a frame dictonary
    frame_dict = {}

    # Preprocess the video
    for video_set in track(videos_to_process, f"Processing Now ..."):
        log.debug(f"************ Start processing {video_set} ************")
        # Create the directory for videos in output root
        os.makedirs(os.path.join(output_root, video_set))
        # Fetch the video set path in data root
        video_set_path = os.path.join(data_root, video_set)
        # Convert the xlsx file into csv format
        xlxs_file_path = os.path.join(video_set_path, f"{video_set}_annotated.xlsx")
        annotation = pd.read_excel(xlxs_file_path, engine="openpyxl")
        # Fetch the start and end frame
        start_frame = annotation.iloc[0]["Frame start"]
        # start_frame = 10
        end_frame = annotation.iloc[-1]["Frame end"]
        # end_frame = 1500
        # Fetch the videos to be processed
        video_files = [
            tmp
            for tmp in os.listdir(video_set_path)
            if bool(re.match(f"{video_set}_part+", tmp))
        ]
        video_files.sort()
        curr_frame = 0
        for idx, video in enumerate(video_files):
            # Open the Video
            cap = cv2.VideoCapture(os.path.join(video_set_path, video))
            # Trim the starting part
            if idx == 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            # Loop the video
            while cap.isOpened() and curr_frame <= end_frame:
                # Read the video
                ret, frame = cap.read()
                if not ret:
                    log.debug(
                        f"Current Frame is {curr_frame} \n Can't receive frame (stream end?). Exiting ..."
                    )
                    break
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                curr_frame += 1
                # Resize the image
                frame = cv2.resize(frame, (int(width), int(height)), cv2.COLOR_RGB2BGR)
                # Save the frame
                frame_path = os.path.join(
                    output_root, video_set, f"{video_set}_{curr_frame}.png"
                )
                cv2.imwrite(frame_path, frame)
                # Search for the corrsponding label
                label_df = annotation[
                    annotation["Frame start"] <= curr_frame + start_frame
                ].iloc[-1]
                frame_step = label_df["Step"]
                frame_task = label_df["Task"]
                frame_dict[curr_frame] = [
                    curr_frame,
                    frame_step,
                    frame_task,
                    frame_path,
                ]

        frame_df = pd.DataFrame.from_dict(
            frame_dict, orient="index", columns=["Frame", "Step", "Task", "Img Path"]
        )
        frame_df.to_csv(
            os.path.join(output_root, video_set, f"{video_set}_annotation.csv")
        )


if __name__ == "__main__":
    main()
