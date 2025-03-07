# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random

import torch
from einops import rearrange
from huggingface_hub import snapshot_download

# from cosmos1.models.autoregressive.nemo.utils import read_input_videos
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.models.autoregressive.nemo.utils import resize_input
from cosmos1.utils import log
import torchvision
import ffmpeg
import glob

TOKENIZER_COMPRESSION_FACTOR = [8, 16, 16]
DATA_RESOLUTION_SUPPORTED = [640, 1024]
NUM_CONTEXT_FRAMES = 33
BOV_TOKEN = 64000
PAD_ID = 64002

def read_input_videos(input_video: str, start_time, chunk_duration_sec) -> torch.tensor:
    """Utility to read the input video and return a torch tensor

    Args:
        input_video (str): A path to .mp4 file
        data_resolution (list, optional): The . Defaults to [640, 1024].

    Returns:
        A torch tensor of the video
    """

    video, _, meta = torchvision.io.read_video(
                input_video, 
                start_pts=start_time, 
                end_pts=start_time + chunk_duration_sec, 
                pts_unit="sec")
    video = video.float() / 255.0
    video = video * 2 - 1

    if video.shape[0] >= NUM_CONTEXT_FRAMES:
        video = video[0:NUM_CONTEXT_FRAMES, :, :, :]
    else:
        log.info(f"Video doesn't have {NUM_CONTEXT_FRAMES} frames. Padding the video with the last frame.")
        # Pad the video
        nframes_in_video = video.shape[0]
        video = torch.cat(
            (video, video[-1, :, :, :].unsqueeze(0).repeat(NUM_CONTEXT_FRAMES - nframes_in_video, 1, 1, 1)),
            dim=0,
        )

    video = video[0:NUM_CONTEXT_FRAMES, :, :, :]
    video = video.permute(0, 3, 1, 2)
    video = resize_input(video, DATA_RESOLUTION_SUPPORTED)
    return video.transpose(0, 1).unsqueeze(0)


def _get_video_tokens(input_path, video_tokenizer, output_dir):
    files = glob.glob(os.path.join(input_path, "*.mp4")) + glob.glob(os.path.join(input_path, "*.webm"))
    if not files:
        raise ValueError(f"Dataset path {input_path} does not contain any .mp4 files.")
    
    for filepath in files:
        
        id = filepath.split("/")[-1].split(".")[0]

        probe = ffmpeg.probe(filepath)
        video_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
        if not video_streams:
            log.info(f"Skipping {filepath} (no video stream found).")
            continue
        
        fps = eval(video_streams["r_frame_rate"]) 
        duration_sec = float(video_streams["duration"]) if "duration" in video_streams else float(probe["format"]["duration"])
        chunk_duration_sec = NUM_CONTEXT_FRAMES / fps  # Convert chunk duration to seconds
        num_chunks = int(duration_sec // chunk_duration_sec)

        print(f"Processing video {filepath} with duration {duration_sec} seconds and {num_chunks} chunks.")

        if duration_sec < chunk_duration_sec:
            log.info(f"Video {filepath} is shorter than {chunk_duration_sec} seconds. Skipped.")
            continue
        

        start_time = 0
        count = 0
        for _ in range(num_chunks):
            input_video = read_input_videos(filepath, start_time=start_time, chunk_duration_sec=chunk_duration_sec).cuda()
            batch_size, channels, frames, height, width = input_video.shape
            latent_shape = (
                (frames - 1) // TOKENIZER_COMPRESSION_FACTOR[0] + 1,
                height // TOKENIZER_COMPRESSION_FACTOR[1],
                width // TOKENIZER_COMPRESSION_FACTOR[2],
            )
            T, H, W = latent_shape
            video_tokenizer.latent_chunk_duration = T
            quantized_out, _ = video_tokenizer.encode(input_video, pixel_chunk_duration=None)
            indices = video_tokenizer.fsq_quantizer.codes_to_indices(quantized_out.permute(0, 2, 3, 4, 1))
            indices = rearrange(indices, "B T H W -> (B T H W)")
            video_tokens = torch.IntTensor([BOV_TOKEN] + indices.tolist() + [PAD_ID] * 64)

            torch.save(video_tokens, f"{output_dir}/video_{count}_{id}.pt")
            count += 1

            #save the video tokens

            start_time = start_time + chunk_duration_sec



def main(args):
  

    if args.encoder_path == "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16":
        args.encoder_path = os.path.join(snapshot_download(args.encoder_path), "encoder.jit")
    if args.decoder_path == "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16":
        args.decoder_path = os.path.join(snapshot_download(args.decoder_path), "decoder.jit")

    video_tokenizer = DiscreteVideoFSQJITTokenizer(
        enc_fp=args.encoder_path,
        dec_fp=args.decoder_path,
        name="discrete_video_fsq",
        pixel_chunk_duration=NUM_CONTEXT_FRAMES,
    ).cuda()


    from pathlib import Path

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _get_video_tokens(args.input_folder, video_tokenizer, args.output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some configurations.")
    
    parser.add_argument(
        "--input_folder",
        required=True,
        type=str,
        help="The path to the a jsonl file. Each line of the file should be a dictionary with two keys. visual_input is a key with the video file as value, and prompt is a key , with the text prompt as value. ",
    )
    parser.add_argument(
        "--encoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to encoder"
    )
    parser.add_argument(
        "--decoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to the decoder"
    )
    parser.add_argument("--split_string", default="4,1,1", type=str, help="The train/test/val split")
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The directory to store the prompt embeddings and video tokens",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
