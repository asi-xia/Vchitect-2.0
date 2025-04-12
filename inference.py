import torch
from models.pipeline import VchitectXLPipeline
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def infer(args):
    pipe = VchitectXLPipeline(args.ckpt_path, device="cuda")
    idx = 0
    '''
    with open(args.test_file,'r') as f:
        for lines in f.readlines():
            for seed in range(5):
    '''
    set_seed(args.seed)
    prompt = args.propmt_text
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        video = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=args.steps, #50,100
            guidance_scale=args.cfg, #7.5
            width=768,
            height=432, #480x288  624x352 432x240 768x432
            frames=args.duration*8 #seconds*frames (default is 8 frames)
        )

    images = video

    from utils import save_as_mp4
    import sys,os
    duration = 1000 / 8

    save_dir = args.save_dir
    os.makedirs(save_dir,exist_ok=True)

    idx += 1
    
    save_as_mp4(images, os.path.join(save_dir, f"zhihui_001")+'.mp4', duration=duration)
                
import sys,os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--propmt_text", type=str)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default='./output')
    parser.add_argument("--ckpt_path", type=str, default='./pretrained_weights')
    args = parser.parse_known_args()[0]
    infer(args)

if __name__ == "__main__":
    main()
