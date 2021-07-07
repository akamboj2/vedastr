"""
What to change between runs:
input size:
    satrn: dummy_samples = torch.randn(1,1,32,100).to('cuda')
    cstr: dummy_samples = torch.randn(1,1,48,192).to('cuda')
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
from vedastr.runners import TestRunner, InferenceRunner
# noqa 402
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
    parser.add_argument('--verbose', default=False, help="print detailed analysis")
    parser.add_argument('--iters', default=100, type=int, help='iters for benchmark')
    parser.add_argument('--batchsize', default=256, help="batch size for the model")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    test_cfg = cfg['test']
    infer_cfg = cfg['inference']
    common_cfg = cfg['common']
    
    
    runner = InferenceRunner(infer_cfg, common_cfg)

#     runner = TestRunner(test_cfg, infer_cfg, common_cfg)
    assert runner.use_gpu, 'Please use gpu for benchmark.'
    runner.load_checkpoint(args.checkpoint)

    model = runner.model
    need_text = runner.need_text
    iters = args.iters
    
    size = cfg['size']
    print("size:",size)
    
    #NOTE: I'm not sure what the right batchsize is here. I would think it could be anything maybe?
    # maybe it shoudl be cfg['batch_max_length'] in config files. it looks like if you print out in test_runner.py
    # they're actually using cfg['samples_per_gpu'] for the tensor sizes they feed into model like (cfg['samples_per_gpu'],1,size[0],size[1])
    dummy_samples = torch.randn(args.batchsize,1,size[0],size[1]).to('cuda')
    dummy_label = torch.ones(args.batchsize,1).to('cuda').long()
    dummy_input = (dummy_samples,dummy_label)
    #print(np.shape(dummy_label))
    
    if need_text:
        model = torch2trt(model, [dummy_input])
    else:
        model = torch2trt(model, [dummy_samples])#,use_onnx=True)
    
    #note: taken from  volksdep benchmark.py
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
    
    if args.verbose:
        with profiler.profile(record_shapes=True,profile_memory=True, use_cuda=True) as prof:
            with profiler.record_function("model_inference"):
                if need_text:
                    pred = model(dummy_input)
                else:
                    pred = model(dummy_samples)            
        #print(prof)
        print('torch.autograd.profiler')
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print("saving chrome trace to: "+cfg_path.split('/')[-1][-3:]+".json")
    #     prof.export_chrome_trace("satrn_trace.json")
        #prof.export_chrome_trace("cstr_trace.json")


if __name__ == '__main__':
    main()

    
### To add actual converter:
"""
import torch.nn.functional as F
import torch.nn as nn
from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test
import collections

@tensorrt_converter('torch.nn.functional.interpolate', enabled=trt_version() >= '7.1')
@tensorrt_converter('torch.nn.functional.upsample', enabled=trt_version() >= '7.1')
def convert_interpolate_trt7(ctx):                                     
    #parse args                     
    input = get_arg(ctx, 'input', pos=0, default=None) 
    size = get_arg(ctx, 'size', pos=1, default=None)
    scale_factor=get_arg(ctx, 'scale_factor', pos=2, default=None)
    mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)

    input_dim = input.dim() - 2
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_resize(input=input_trt)

    shape = size
    if shape != None:
        if isinstance(shape, collections.Sequence):
           shape  = [input.size(1)] + list(shape)
        else:
            shape = [input.size(1)] + [shape] * input_dim

        layer.shape = shape

    scales = scale_factor
    if scales != None:
        if not isinstance(scales, collections.Sequence):
            scales = [scales] * input_dim
        layer.scales = [1] + list(scales)

    resize_mode = mode
    if resize_mode.lower() in ["linear","bilinear","trilinear"]:
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode=trt.ResizeMode.NEAREST

    if align_corners != None:
        #ak  
        #layer.align_corners = align_corners
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        #  https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html#iresizelayer
        layer.coordinate_transformation = trt.ResizeCoordinateTransformationDoc.ALIGN_CORNERS 

    output._trt = layer.get_output(0)
"""