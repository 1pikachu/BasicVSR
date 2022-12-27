# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import time

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def parse_args():
    # modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('--config', default="configs/basicvsr_plusplus_reds4.py", help='test config file path')
    parser.add_argument('--checkpoint', default="/home2/pytorch-broad-models/BasicVSR/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth", help='checkpoint file')
    parser.add_argument('--input_dir', default="/home2/pytorch-broad-models/BasicVSR/_2ynx9GQIsA_000001_000011.mp4", help='directory of the input video')
    parser.add_argument('--output_dir', default="results/demo_000", help='directory of the output video')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template of the file names')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    args = parser.parse_args()
    # only 1
    args.batch_size = 1
    # must float32
    args.precision = "float32"
    print(args)
    return args

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def inference(args):
    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    total_time = 0.0
    total_sample = 0

    if args.profile and args.device == "xpu":
        for i in range(args.num_iter + args.num_warmup):
            elapsed = time.time()
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                output = restoration_video_inference(args, args.model, args.input_dir,
                                                     args.window_size, args.start_idx,
                                                     args.filename_tmpl, args.max_seq_len, i)
            torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == int((args.num_iter + args.num_warmup)/2):
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
                torch.save(prof.table(sort_by="id", row_limit=100000),
                    timeline_dir+'profile_detail_withId.pt')
                prof.export_chrome_trace(timeline_dir+"trace.json")
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                elapsed = time.time()
                with torch.jit.fuser(fuser_mode):
                    output = restoration_video_inference(args, args.model, args.input_dir,
                                                         args.window_size, args.start_idx,
                                                         args.filename_tmpl, args.max_seq_len, i)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                elapsed = time.time()
                output = restoration_video_inference(args, args.model, args.input_dir,
                                                     args.window_size, args.start_idx,
                                                     args.filename_tmpl, args.max_seq_len, i)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i in range(args.num_iter + args.num_warmup):
            elapsed = time.time()
            with torch.jit.fuser(fuser_mode):
                output = restoration_video_inference(args, args.model, args.input_dir,
                                                     args.window_size, args.start_idx,
                                                     args.filename_tmpl, args.max_seq_len, i)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for i in range(args.num_iter + args.num_warmup):
            elapsed = time.time()
            output = restoration_video_inference(args, args.model, args.input_dir,
                                                 args.window_size, args.start_idx,
                                                 args.filename_tmpl, args.max_seq_len, i)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

def main():
    """ Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir'
    is set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    args.model = init_model(
        args.config, args.checkpoint, device=torch.device(args.device))

    if args.channels_last:
        try:
            args.model = args.model.to(memory_format=torch.channels_last)
            print("---- use NHWC format")
        except RuntimeError as e:
            print("---- use normal format")
            print("fail to enable channels_last=1: ", e)

    # 1iter forward
    with torch.no_grad():
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                inference(args)
        elif args.precision == "float16" and args.device == "xpu":
            # optimize
            model = torch.xpu.optimize(model=model, dtype=torch.float16)
            print("---- xpu optimize")
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                inference(args)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                inference(args)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                inference(args)
        else:
            print("---- no autocast")
            inference(args)

    """ disable
    file_extension = os.path.splitext(args.output_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:  # save as video
        h, w = output.shape[-2:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_dir, fourcc, 25, (w, h))
        for i in range(0, output.size(1)):
            img = tensor2img(output[:, i, :, :, :])
            video_writer.write(img.astype(np.uint8))
        cv2.destroyAllWindows()
        video_writer.release()
    else:
        for i in range(args.start_idx, args.start_idx + output.size(1)):
            output_i = output[:, i - args.start_idx, :, :, :]
            output_i = tensor2img(output_i)
            save_path_i = f'{args.output_dir}/{args.filename_tmpl.format(i)}'

            mmcv.imwrite(output_i, save_path_i)
    """


if __name__ == '__main__':
    main()
