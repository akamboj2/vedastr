import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2  # noqa 402

from vedastr.runners import InferenceRunner  # noqa 402
from vedastr.utils import Config  # noqa 402

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('image', type=str, help='input image path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    if os.path.isfile(args.image):
        images = [args.image]
    else:
        images = [
            os.path.join(args.image, name) for name in filter(lambda x: False if x[-18:]=='.ipynb_checkpoints' else True, os.listdir(args.image))
        ]
        print(images)
    for img in images:
        image = cv2.imread(img)
        print(np.shape(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("save_img.png", image)
        print("in inference.py image size", np.shape(image))
        pred_str, probs = runner(image)
        runner.logger.info('Text in {} is:\t {} '.format(img, pred_str))


if __name__ == '__main__':
    main()
