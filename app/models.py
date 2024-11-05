from pydantic import BaseModel
from typing import Union, List

class GenerateRequest(BaseModel):
    prompt: Union[str, List[str]]
    num_inference_steps: int
    seed: int
    cfg: float
    save_disk_path: str = None