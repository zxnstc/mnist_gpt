from pathlib import Path
import time
import os
import inspect
import torch
from loguru import logger


starttime = time.strftime("%Y-%m-%d_%H:%M:%S")

def get_file_name():
    frame = inspect.currentframe()
    while frame.f_back:
        frame = frame.f_back
    main_script_name = frame.f_globals['__file__']
    filename=os.path.basename(main_script_name)
    parts = filename.split('.', 1)
    return parts[0]

filename=get_file_name()




def get_exp_id() -> str:
    return f"{filename}/{starttime[5:16]}"

def save_checkpoints(state:dict,iter:int):
    exp_id=get_exp_id()
    parent_path=Path("/data/mnist_gpt/checkpoints/")
    ckp_dir=parent_path/exp_id

    ckp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[checkpoint] Saving checkpoint")

    torch.save(state,ckp_dir/f"{iter:05d}.pt")
    logger.info(f"Checkpoint saved to {ckp_dir}")
    return ckp_dir


