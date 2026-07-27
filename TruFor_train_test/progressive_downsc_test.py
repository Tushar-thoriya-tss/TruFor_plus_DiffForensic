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

Progressive Downscale OOM fallback:
    On CUDA OOM, retries inference at 75% → 50% → max-dim-512 of original size.
"""

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.nn import functional as F

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from lib.config import config, update_config
from lib.utils import get_model
from dataset.dataset_test import TestDataset

parser = argparse.ArgumentParser(description='Test TruFor - Progressive Downscale')
parser.add_argument('-g',   '--gpu',     type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in',  '--input',   type=str, default='../images', help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output',  type=str, default='../output', help='output folder')
parser.add_argument('-exp', '--experiment', type=str, default='trufor_ph3')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input   = args.input
output  = args.output
gpu     = args.gpu
save_np = args.save_np

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
print(f'Device: {device}')

if device != 'cpu':
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

if '*' in input:
    list_img = glob(input, recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
elif os.path.isfile(input):
    list_img = [input]
elif os.path.isdir(input):
    list_img = glob(os.path.join(input, '**/*'), recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
else:
    raise ValueError("input is neither a file or a folder")

print(f'Image Verifyer is Running.....')

from PIL import Image
valid_images = []
for img_path in list_img:
    try:
        if os.path.getsize(img_path) > 0:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                im.convert("RGB")
            valid_images.append(img_path)
    except:
        pass
list_img = valid_images
print(f'Image Verification Completed!')
print(f'Total Number of Valid Images: {len(list_img)}')


def run_model_progressive(model, rgb, save_np):
    """
    Try inference at progressively lower resolutions on CUDA OOM.
    Scales tried: 100% → 75% → 50% → max-dim-512 fallback.
    Returns: (pred, conf, det, npp, rgb_used)
        pred, conf, det, npp : raw model outputs (tensors or None)
        rgb_used             : the tensor that succeeded (needed for imgsize fallback)
    """
    h_orig, w_orig = rgb.shape[2], rgb.shape[3]

    # Progressively smaller scales to try before the hard 512 fallback
    scales = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.5]

    for scale in scales:
        try:
            if scale < 1.0:
                print(f"Trying RESIZING IMAGE.... at {scale}")
                torch.cuda.empty_cache()
                new_h = max(1, int(h_orig * scale))
                new_w = max(1, int(w_orig * scale))
                rgb_try = F.interpolate(rgb, size=(new_h, new_w),
                                        mode='bilinear', align_corners=False)
                print(f'  [OOM] Retrying at {int(scale*100)}% '
                      f'({new_h}x{new_w})...')
                print(f"Origianl Image size was: (height) and (weight): {h_orig}x{w_orig}")
            else:
                rgb_try = rgb

            pred, conf, det, npp = model(rgb_try, save_np=save_np)
            return pred, conf, det, npp, rgb_try

        except torch.cuda.OutOfMemoryError:
            if scale == scales[-1]:
                # All percentage scales failed — try the hard 512 cap next
                pass
            continue

    # Final fallback: cap the longest side at 512
    torch.cuda.empty_cache()
    scale_512 = 512 / max(h_orig, w_orig)
    new_h = max(1, int(h_orig * scale_512))
    new_w = max(1, int(w_orig * scale_512))
    rgb_try = F.interpolate(rgb, size=(new_h, new_w),
                            mode='bilinear', align_corners=False)
    print(f'  [OOM] Final fallback at max-dim-512 ({new_h}x{new_w})...')
    pred, conf, det, npp = model(rgb_try, save_np=save_np)
    return pred, conf, det, npp, rgb_try


test_dataset = TestDataset(list_img=list_img)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1)   # 1 to allow arbitrary input sizes

if config.TEST.MODEL_FILE:
    model_state_file = config.TEST.MODEL_FILE
else:
    raise ValueError("Model file is not specified.")

print('=> loading model from {}'.format(model_state_file))
checkpoint = torch.load(model_state_file, map_location=torch.device(device), weights_only=False)
print("Epoch: {}".format(checkpoint['epoch']))

model = get_model(config)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

with torch.inference_mode():
    for index, (rgb, path) in enumerate(tqdm(testloader)):
        if os.path.splitext(os.path.basename(output))[1] == '':  # output is a directory
            path = path[0]
            root = input.split('*')[0]

            if os.path.isfile(input):
                sub_path = path.replace(os.path.dirname(root), '').strip()
            else:
                sub_path = path.replace(root, '').strip()

            if sub_path.startswith('/'):
                sub_path = sub_path[1:]

            filename_out = os.path.join(output, sub_path) + '.npz'
        else:  # output is a filename
            filename_out = output

        if not filename_out.endswith('.npz'):
            filename_out = filename_out + '.npz'

        # by default it does not overwrite
        if not (os.path.isfile(filename_out)):
            try:
                rgb = rgb.to(device)
                model.eval()

                det  = None
                conf = None

                # -------------------------------------------------------------
                # Progressive downscale: 100% → 75% → 50% → max-dim-512
                # -------------------------------------------------------------
                orig_imgsize = tuple(rgb.shape[2:])  # preserve original size
                pred, conf, det, npp, rgb_used = run_model_progressive(model, rgb, save_np)

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

                out_dict = dict()
                out_dict['map'    ] = pred
                out_dict['imgsize'] = orig_imgsize   # always save original size
                if det is not None:
                    out_dict['score'] = det_sig
                if conf is not None:
                    out_dict['conf'] = conf
                if save_np:
                    out_dict['np++'] = npp

                from os import makedirs
                makedirs(os.path.dirname(filename_out), exist_ok=True)
                np.savez(filename_out, **out_dict)
            except:
                import traceback
                traceback.print_exc()
                pass
