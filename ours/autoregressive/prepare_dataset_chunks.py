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

import os
from argparse import ArgumentParser
from glob import glob

import torch
from einops import rearrange
from huggingface_hub import snapshot_download
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

from cosmos1.models.autoregressive.nemo.utils import resize_input
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.utils import log
import torchvision
import ffmpeg

TOKENIZER_COMPRESSION_FACTOR = [8, 16, 16]
DATA_RESOLUTION_SUPPORTED = [640, 1024]
NUM_CONTEXT_FRAMES = 49

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

    builders = {}
    key = "text"
    filepaths_final = glob(f"{args.input_videos_dir}/*.mp4") + glob(f"{args.input_videos_dir}/*.webm")

    for filepath in filepaths_final:
        video_basename = os.path.splitext(os.path.basename(filepath))[0]
        builders[key] = indexed_dataset.make_builder(
            f"{args.output_prefix}/{video_basename}.bin",
            impl="mmap",
            chunk_size=64,
            pad_id=0,
            retrieval_db=None,
            vocab_size=64000,
            stride=64,
        )

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
            indices = rearrange(indices, "B T H W -> (B T H W)").detach().cpu()
            builders[key].add_item(torch.IntTensor(indices).detach().cpu())
            

        
            start_time = start_time + chunk_duration_sec
        builders[key].end_document()
        builders[key].finalize(
                f"{args.output_prefix}/{video_basename}.idx",
            )


    log.info(f"Stored the .bin and .idx files in {args.output_prefix}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_videos_dir", required=True, type=str, help="The path to the input videos")
    parser.add_argument(
        "--encoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to encoder"
    )
    parser.add_argument(
        "--decoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to the decoder"
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        type=str,
        help="The directory along with the output file name to write the .idx and .bin files (e.g /path/to/output/sample)",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
