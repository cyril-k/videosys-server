import os
import logging
import time

from videosys import VideoSysEngine


logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def generate_video(engine: VideoSysEngine, prompt: str, guidance_scale: float, num_inference_steps: int, seed: int, save_disk_path: str):

    if save_disk_path and not os.path.isdir(save_disk_path):
        default_path = os.path.join(os.path.expanduser("~"), "output")
        os.makedirs(default_path, exist_ok=True)
        logger.warning(f"Invalid save_disk_path. Using default path: {default_path}")
        save_disk_path = default_path

    start_time = time.perf_counter()
    videos = engine.generate(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        seed=seed,
    ).video
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    return elapsed_time, videos