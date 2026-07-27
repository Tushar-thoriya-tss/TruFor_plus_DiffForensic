# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro

Progressive Downscale OOM fallback (v2 - clean rewrite):
    Faithful replication of test.py with progressive downscale on CUDA OOM.
    On OOM, retries inference at 95% -> 90% -> 85% -> 80% -> 75% -> 50%
    -> max-dim-512 of original size.

    Fixes over v1 (progressive_downsc_test.py):
    - Atomic write: saves to a .tmp.npz file first, then os.replace() to final
      path, preventing corrupt/partial .npz files on interruption.
    - On any exception: removes the output file (and temp file) so the
      image is retried on the next run instead of being skipped forever.
    - orig_imgsize is captured from the original tensor BEFORE any resize,
      so 'imgsize' in the .npz always reflects the true input image size.
    - All imports moved to the top of the file (none inside the loop).
"""

import gc
import os
import sys
import argparse
import traceback
import numpy as np
from os import makedirs
from tqdm import tqdm
from glob import glob

import torch
from torch.nn import functional as F
from PIL import Image

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from lib.config import config, update_config
from lib.utils import get_model
from dataset.dataset_test import TestDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Test TruFor - Progressive Downscale v2')
parser.add_argument('-g',      '--gpu',        type=int,  default=0,           help='device, use -1 for cpu')
parser.add_argument('-in',     '--input',      type=str,  default='../images',  help='single file, directory, or glob')
parser.add_argument('-out',    '--output',     type=str,  default='../output',  help='output folder')
parser.add_argument('-exp',    '--experiment', type=str,  default='trufor_ph3')
parser.add_argument('-save_np','--save_np',    action='store_true',             help='save Noiseprint++ map')
parser.add_argument('opts', help='other options', default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input_path = args.input   # renamed from 'input' to avoid shadowing built-in
output     = args.output
gpu        = args.gpu
save_np    = args.save_np

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
print(f'Device: {device}')

if device != 'cpu':
    import torch.backends.cudnn as cudnn
    cudnn.benchmark     = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled       = config.CUDNN.ENABLED


# ---------------------------------------------------------------------------
# Build image list
# ---------------------------------------------------------------------------
if '*' in input_path:
    list_img = glob(input_path, recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
elif os.path.isfile(input_path):
    list_img = [input_path]
elif os.path.isdir(input_path):
    list_img = glob(os.path.join(input_path, '**/*'), recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
else:
    raise ValueError("input is neither a file nor a folder")


# ---------------------------------------------------------------------------
# Image verification  (same as original test.py)
# ---------------------------------------------------------------------------
# print('Image Verifier is Running.....')
# valid_images = []
# for img_path in list_img:
#     try:
#         if os.path.getsize(img_path) > 0:
#             with Image.open(img_path) as im:
#                 im.verify()                 # basic integrity check
#             with Image.open(img_path) as im:    # takes extra time but ensures full decode
#                 im.convert("RGB")
#             valid_images.append(img_path)
#     except Exception:
#         pass                                # silently skip unreadable files
# list_img = valid_images
# print('Image Verification Completed!')
# print(f'Total Number of Valid Images: {len(list_img)}')


if os.path.exists('verified_files_catch.txt'):
    print('verified_files_catch.txt Loaded....!')
    with open('verified_files_catch.txt', 'r') as f:
        valid_images = f.read()
        list_img = valid_images.split('\n')
else: 
    print('Image Verifier is Running.....')
    valid_images = []
    for img_path in tqdm(list_img, desc='Image Verifier'):
        try:
            if os.path.getsize(img_path) > 0:
                with Image.open(img_path) as im:
                    im.verify()                 # basic integrity check
                with Image.open(img_path) as im:    # takes extra time but ensures full decode
                    im.convert("RGB")
                valid_images.append(img_path)
        except Exception:
            pass                                # silently skip unreadable files
    list_img = valid_images
    print('Image Verification Completed!')
    print(f'Total Number of Valid Images: {len(list_img)}')
    with open('verified_files_catch.txt', 'w') as f:
        for i in list_img:
            f.write(i + '\n')
    print("Saved...!: verified_files_catch.txt")

# ---------------------------------------------------------------------------
# Progressive downscale helper
# ---------------------------------------------------------------------------
def run_model_progressive(model, rgb, save_np_flag):
    """
    Run model inference; on CUDA OOM retry at progressively smaller sizes.

    Scales tried: 100% -> 95% -> 90% -> 85% -> 80% -> 75% -> 50%
                  -> final fallback: longest side capped at 512 px.

    Returns
    -------
    pred, conf, det, npp : raw model output tensors (some may be None)
    orig_imgsize         : (H, W) of the original (un-resized) tensor,
                           captured here so callers always have the true size.
    """
    h_orig, w_orig = rgb.shape[2], rgb.shape[3]
    orig_imgsize   = (h_orig, w_orig)          # capture BEFORE any resize

    scales = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.5]

    for scale in scales:
        try:
            if scale < 1.0:
                torch.cuda.empty_cache()
                new_h = max(1, int(h_orig * scale))
                new_w = max(1, int(w_orig * scale))
                rgb_try = F.interpolate(rgb, size=(new_h, new_w),
                                        mode='bilinear', align_corners=False)
                print(f'  [OOM] Retrying at {int(scale * 100)}% '
                      f'({new_h}x{new_w})  [original: {h_orig}x{w_orig}]')
            else:
                rgb_try = rgb

            pred, conf, det, npp = model(rgb_try, save_np=save_np_flag)
            return pred, conf, det, npp, orig_imgsize

        except torch.cuda.OutOfMemoryError:
            continue   # try next scale

    # All percentage scales exhausted -> hard cap at longest side = 512
    torch.cuda.empty_cache()
    scale_512 = 512.0 / max(h_orig, w_orig)
    new_h = max(1, int(h_orig * scale_512))
    new_w = max(1, int(w_orig * scale_512))
    rgb_try = F.interpolate(rgb, size=(new_h, new_w),
                            mode='bilinear', align_corners=False)
    print(f'  [OOM] Final fallback: max-dim-512 ({new_h}x{new_w})  '
          f'[original: {h_orig}x{w_orig}]')
    pred, conf, det, npp = model(rgb_try, save_np=save_np_flag)
    return pred, conf, det, npp, orig_imgsize


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
test_dataset = TestDataset(list_img=list_img)
testloader   = torch.utils.data.DataLoader(test_dataset, batch_size=1)


# ---------------------------------------------------------------------------
# Load model  (identical to original test.py)
# ---------------------------------------------------------------------------
if config.TEST.MODEL_FILE:
    model_state_file = config.TEST.MODEL_FILE
else:
    raise ValueError("Model file is not specified.")

print('=> loading model from {}'.format(model_state_file))
checkpoint = torch.load(model_state_file,
                        map_location=torch.device(device),
                        weights_only=False)
print("Epoch: {}".format(checkpoint['epoch']))

model = get_model(config)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
with torch.inference_mode():
    for index, (rgb, path) in enumerate(tqdm(testloader)):

        # ── Build output file path (identical logic to original test.py) ──
        if os.path.splitext(os.path.basename(output))[1] == '':  # output is a directory
            path = path[0]
            root = input_path.split('*')[0]

            if os.path.isfile(input_path):
                sub_path = path.replace(os.path.dirname(root), '').strip()
            else:
                sub_path = path.replace(root, '').strip()

            if sub_path.startswith('/'):
                sub_path = sub_path[1:]

            filename_out = os.path.join(output, sub_path) + '.npz'
        else:                                                      # output is a filename
            filename_out = output

        if not filename_out.endswith('.npz'):
            filename_out = filename_out + '.npz'

        # Skip if a valid (complete) output already exists
        if os.path.isfile(filename_out):
            continue

        # ── Temporary file path used for atomic write ──
        # NOTE: must end in .npz so numpy does NOT auto-append .npz again,
        #       otherwise os.replace() would fail (file.npz.tmp vs file.npz.tmp.npz).
        tmp_out = filename_out[:-4] + '.tmp.npz'   # e.g. file.tmp.npz -> file.npz

        try:
            rgb = rgb.to(device)
            model.eval()

            det  = None
            conf = None

            # ── Progressive downscale inference ──
            pred, conf, det, npp, orig_imgsize = run_model_progressive(
                model, rgb, save_np)

            # ── Post-process outputs (identical to original test.py) ──
            if conf is not None:
                conf = torch.squeeze(conf, 0)
                conf = torch.sigmoid(conf)[0]
                conf = conf.cpu().numpy()

            if npp is not None:
                npp = torch.squeeze(npp, 0)[0]
                npp = npp.cpu().numpy()

            if det is not None:
                det_sig = torch.sigmoid(det).item()

            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            # ── Build output dict ──
            out_dict = {}
            out_dict['map']     = pred
            out_dict['imgsize'] = orig_imgsize   # always the ORIGINAL image size
            if det is not None:
                out_dict['score'] = det_sig
            if conf is not None:
                out_dict['conf']  = conf
            if save_np:
                out_dict['np++']  = npp

            # ── Atomic write: .tmp.npz -> final .npz ──
            makedirs(os.path.dirname(filename_out), exist_ok=True)
            np.savez(tmp_out, **out_dict)           # write to temp first
            os.replace(tmp_out, filename_out)       # atomic rename on Linux

        except Exception:
            traceback.print_exc()
            # Remove any partial output so the image is retried on next run
            for leftover in (filename_out, tmp_out):
                if os.path.isfile(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass
        finally:
            # Always free GPU memory between images
            torch.cuda.empty_cache()
            gc.collect()
