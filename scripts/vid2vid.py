import numpy as np
import streamlit as st
import cv2
import subprocess

import torch
from PIL import Image

from main import style_transfer_image_from_prompt
from stable_diffusion_server.utils import log_time

video_data = st.file_uploader("Upload file", ['mp4', 'mov', 'avi'])

temp_file_to_save = './tmp1.mp4'
temp_file_result = './tmp2.mp4'


# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet.
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

input_text = st.text_input('Enter a prompt', 'a magical elf in a forest')
with torch.inference_mode():

    if video_data:
        write_bytesio_to_file(temp_file_to_save, video_data)

        # so now we can process it with OpenCV functions
        cap = cv2.VideoCapture(temp_file_to_save)

        # grab some parameters of video to use them for writing a new, processed video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int
        st.write(width, height, frame_fps)

        # specify a writer to write a processed video to a disk frame by frame
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height), isColor=True)

        while True:
            ret, frame = cap.read()
            if not ret: break
            # use sdif to get a frame
            image_pil = Image.fromarray(frame)
            imagepilresult = style_transfer_image_from_prompt(input_text, None, 0.6, True, image_pil)
            # imagepilresult.save("teststyletransfer-cnet-tmp.png")
            # convert result to opencv frame
            # todo remove pil conversion
            with log_time('pilocv'):
                ocvframe = cv2.cvtColor(np.array(imagepilresult), cv2.COLOR_RGB2BGR)
            out_mp4.write(ocvframe)

        ## Close video files
        out_mp4.release()
        cap.release()

        ## Reencodes video to H264 using ffmpeg
        ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
        ##  ... and will probably fail in streamlit cloud
        st.markdown(
            f"Download high res processed video: <a href=\"/app/{temp_file_result}\" download>Click to Download</a>",
            unsafe_allow_html=True)

        convertedVideo = "./tmp2.mp4"
        subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))

        converted_video = temp_file_result + ".webm"
        subprocess.call(
            args=f"/home/lee/code/ffmpeg-git-20231006-amd64-static/ffmpeg -y -i {temp_file_result} -vcodec libvpx -acodec libvorbis {converted_video}".split(
                " "))

        ## Show results
        col1, col2 = st.columns(2)
        col1.header("Original Video")
        col1.video(temp_file_to_save)
        col2.header("Output from OpenCV (MPEG-4)")
        col2.video(temp_file_result)
        col2.header("After conversion to H264")
        col2.video(convertedVideo)
