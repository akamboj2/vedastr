"""
What to change between runs:
input size:
    cstr: dummy_samples = torch.randn(1,1,48,192).to('cuda')
    satrn: dummy_samples = torch.randn(1,1,32,100).to('cuda')
json file name:
    prof.export_chrome_trace("satrn_trace.json")
    prof.export_chrome_trace("cstr_trace.json")


"""
#CSTR benchmark
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import cv2  # noqa 402
from volksdep.benchmark import benchmark  # noqa 402

from tools.deploy.utils import CALIBRATORS, CalibDataset, Metric, MetricDataset  # noqa 402
from vedastr.runners import TestRunner  # noqa 402
from vedastr.utils import Config  # noqa 402

import numpy as np
import torch

import time
from torch2trt import torch2trt
import torch.autograd.profiler as profiler


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('image', type=str, help='sample image path')
    parser.add_argument(
        '--dtypes',
        default=('fp32', 'fp16', 'int8'),
        nargs='+',
        type=str,
        choices=['fp32', 'fp16', 'int8'],
        help='dtypes for benchmark')
    parser.add_argument(
        '--iters', default=100, type=int, help='iters for benchmark')
    parser.add_argument(
        '--calibration_images',
        default=None,
        type=str,
        help='images dir used when int8 in dtypes')
    parser.add_argument(
        '--calibration_modes',
        nargs='+',
        default=['entropy', 'entropy_2', 'minmax'],
        type=str,
        choices=['entropy_2', 'entropy', 'minmax'],
        help='calibration modes for benchmark')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    test_cfg = cfg['test']
    infer_cfg = cfg['inference']
    common_cfg = cfg['common']

    runner = TestRunner(test_cfg, infer_cfg, common_cfg)
    assert runner.use_gpu, 'Please use gpu for benchmark.'
    runner.load_checkpoint(args.checkpoint)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aug = runner.transform(image=image, label='')
    image, dummy_label = aug['image'], aug['label']  # noqa 841
    image = image.unsqueeze(0)
    input_len = runner.converter.test_encode(1)[0]
    model = runner.model
    need_text = runner.need_text
    if need_text:
        shape = tuple(image.shape), tuple(input_len.shape)
    else:
        shape = tuple(image.shape)

    dtypes = args.dtypes
    iters = args.iters
    int8_calibrator = None
    if args.calibration_images:
        calib_dataset = CalibDataset(args.calibration_images, runner.converter,
                                     runner.transform, need_text)
        int8_calibrator = [
            CALIBRATORS[mode](dataset=calib_dataset)
            for mode in args.calibration_modes
        ]
#     dataset = runner.test_dataloader.dataset
    dataset = runner.dataset[0] #list(runner.test_dataloader.values())[0]
#     print(type(dataset))
#     print(len(dataset))
#     for l in dataset:
#         print(l)
#         break
    
    dataset = MetricDataset(dataset, runner.converter, need_text)
    metric = Metric(runner.metric, runner.converter)
#     benchmark(
#         model,
#         shape,
#         dtypes=dtypes,
#         iters=iters,
#         int8_calibrator=int8_calibrator,
#         dataset=dataset,
#         metric=metric)
    """Note this call to benchmark doesn't work because the type of the target/label in dataset is a
    string not an int or list or tuple
    like when you do input,label=dataset[0] label is a string! breaks on line 57 of volksdep/utils.py and line 45 of volksdep/benchmark.py!
    I feel like i shouldn't change the structue of dataset because that would make train not work but ideally i think i should just be able to cast the string to a numpy array (or put it in a numpy array)"""

    
    #note taken from benchmark.py
    dummy_samples = torch.randn(1,1,32,100).to('cuda')
    #dummy_samples = torch.randn(1,1,48,192).to('cuda')

    dummy_label = torch.ones(1,1).to('cuda').long()
    dummy_input = (dummy_samples,dummy_label)
    print(np.shape(dummy_label))
    
    #model = torch2trt(model, dummy_samples)
    
    torch.cuda.profiler.init('cstr_trt.nvvp',output_mode='csv')
    
    # collect region using context manager
    torch.cuda.current_stream().synchronize() 
    with torch.cuda.profiler.profile():
        y = model(dummy_input)
        torch.cuda.current_stream().synchronize()
        
        # collect region using start/stop
#     torch.cuda.current_stream().synchronize()
#     torch.cuda.profiler.start()
#     y = model_trt(x)
#     torch.cuda.current_stream().synchronize()
#     torch.cuda.profiler.stop()
    

    
    print('done with profiler profiling')
    #print(torch.cuda.profiler.table())
    
    with torch.no_grad():
            # warm up
            for _ in range(10):
                model(dummy_input)

            # throughput evaluate
            torch.cuda.current_stream().synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(dummy_input)
            torch.cuda.current_stream().synchronize()
            t1 = time.time()
            throughput = 1.0 * iters / (t1 - t0)

            # latency evaluate
            torch.cuda.current_stream().synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(dummy_input)
                torch.cuda.current_stream().synchronize()
            t1 = time.time()
            latency = round(1000.0 * (t1 - t0) / iters, 2)
    print("throughput,latency",throughput,latency)
    
    with profiler.profile(record_shapes=True,profile_memory=True, use_cuda=True) as prof:
        with profiler.record_function("model_inference"):
            if need_text:
                pred = model(dummy_input)
            else:
                pred = model(dummy_samples)            
    #print(prof)
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    prof.export_chrome_trace("satrn_trace.json")
    #prof.export_chrome_trace("cstr_trace.json")


if __name__ == '__main__':
    main()
