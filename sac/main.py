import sys
import os
from os.path import dirname, join 
from shutil import copy2

from scripts.train import train
from scripts.eval import eval

VERBOSE = True


if __name__ == "__main__":
    model_path = None
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_file.yaml>")
        sys.exit(1)
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    config_file = sys.argv[1]
    config_stem = os.path.splitext(config_file)[0]
    config_path = join(dirname(__file__), "config/", config_file)
    if VERBOSE:
        print(f"loading config file '{config_path}'.")

    # prepare file structure
    dest_dir = join(dirname(__file__), "results/", config_stem) 
    if VERBOSE:
        print(f"Creating destination result directory '{dest_dir}'.")
    os.makedirs(dest_dir, exist_ok=True)
    copy2(config_path, dest_dir) # copy config file to be available in result directory

    # this script will save the model under 'results/<name_of_config_file>/{+details}>.zip' 
    train(config_path=config_path, dest_path=dest_dir, model_path=model_path)
    eval(config_path=config_path, dest_path=dest_dir)

    print(f"Model according to config file '{config_file}' trained and saved to '{dest_dir}'.")