import os
import argparse
import torch
import core

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("mode", help = "Select the mode", type = str)
    parser.add_argument("--scale", default = 2, type = int,
                        help="Upscale factor")
    parser.add_argument("--device", default = "cuda:0", type = str,
                        help="Choose your device you want yo use")
    parser.add_argument("--format", default = None, type = str,
                        help="Output format")
    args = parser.parse_args()
    return args

def demo():
    args = parse_args()
    d = core.run_demo(args)
    d.run()

if __name__ == "__main__":
    demo()