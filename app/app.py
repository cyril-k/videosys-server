import os
import io
import logging
import imageio

import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from videosys import CogVideoXConfig, VideoSysEngine


from models import GenerateRequest
from inference import generate_video

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# random.seed(42)
# Environment setup for NCCL
# os.environ["NCCL_BLOCKING_WAIT"] = "1"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

initialized = False
engine = None

app = FastAPI()

def initialize_engine():
    global engine, initialized
    served_model = os.environ.get("SERVED_MODEL", "THUDM/CogVideoX-5b")

    config = CogVideoXConfig(
        served_model, 
        num_gpus=torch.cuda.device_count(), 
        enable_pab=True
    )

    engine = VideoSysEngine(config)
    logger.info("========= Initialized inference pipeline =========")
    initialized = True

@app.on_event("startup")
async def startup_event():
    initialize_engine()


@app.get("/initialize")
async def check_initialize():
    global initialized
    if initialized:
        return {"status": "initialized"}
    else:
        return {"status": "initializing"}, 202

def images_to_mp4_bytes(images, fps=24):
    video_buffer = io.BytesIO()

    with imageio.get_writer(video_buffer, format='mp4', mode='I', fps=fps) as writer:
        for image in images:
            np_frame = np.array(image)
            writer.append_data(np_frame)

    video_buffer.seek(0)
    return video_buffer

@app.post("/generate")
async def generate(request: GenerateRequest):
    global engine, initialized
    logger.info("Received POST request for image generation")
    prompt = request.prompt
    num_inference_steps = request.num_inference_steps
    seed = request.seed
    cfg = request.cfg
    save_disk_path = request.save_disk_path

    elapsed_time, videos = generate_video(
        engine=engine,
        prompt=prompt,
        guidance_scale=cfg,
        num_inference_steps=num_inference_steps,
        seed=seed,
        save_disk_path=save_disk_path,
    )
    video = videos[0]

    if save_disk_path:
        output_base64 = ""
        engine.save_video(video, f"{save_disk_path}/{prompt}.mp4")
        return {
            "message": "Image generated successfully",
            "elapsed_time": str(elapsed_time),
            "output": output_base64 if not save_disk_path else save_disk_path,
            "save_to_disk": save_disk_path is not None,
        }
    else:
        bytestream = images_to_mp4_bytes(video, fps=8)
        print(bytestream)
        headers = {
            "X-Video-Title": prompt,
            "X-Video-Description": "This is a dynamically generated video.",
            "X-Video-Generation-Duration": str(elapsed_time)
        }

        return StreamingResponse(bytestream, media_type="video/mp4", headers=headers)