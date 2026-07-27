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

Tiled / Sliding Window Inference:
    Splits large images into overlapping 512x512 tiles, runs inference on
    each tile independently, then stitches results back to full resolution
    by averaging overlapping regions.  Full image resolution is preserved.
"""

import gc
import sys
import os
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

parser = argparse.ArgumentParser(description='Test TruFor - Tiled Inference')
parser.add_argument('-g',   '--gpu',     type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in',  '--input',   type=str, default='../images', help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output',  type=str, default='../output', help='output folder')
parser.add_argument('-exp', '--experiment', type=str, default='trufor_ph3')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('-tile_size', '--tile_size', type=int, default=512,
                    help='tile size for sliding window inference (default: 512)')
parser.add_argument('-overlap',   '--overlap',   type=int, default=64,
                    help='overlap between adjacent tiles in pixels (default: 64)')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input_path = args.input   # renamed from 'input' to avoid shadowing built-in
output     = args.output
gpu        = args.gpu
save_np    = args.save_np
TILE_SIZE  = args.tile_size
OVERLAP    = args.overlap

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
print(f'Device: {device}')
print(f'Tile size: {TILE_SIZE}  |  Overlap: {OVERLAP}')

if device != 'cpu':
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

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

print(f'Image Verifier is Running.....')

valid_images = []
for img_path in list_img:
    try:
        if os.path.getsize(img_path) > 0:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                im.convert("RGB")
            valid_images.append(img_path)
    except Exception:
        pass
list_img = valid_images
print(f'Image Verification Completed!')
print(f'Total Number of Valid Images: {len(list_img)}')


# ---------------------------------------------------------------------------
# Helper: generate tile start positions that fully cover [0, size)
# ---------------------------------------------------------------------------
def _tile_starts(size, tile_size, stride):
    """
    Return a sorted, deduplicated list of start positions for tiles of
    `tile_size` with `stride` step that completely covers [0, size).
    The last position is always max(0, size - tile_size) so the final
    tile ends exactly at `size`.
    """
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size, stride))
    starts.append(size - tile_size)        # ensure the tail is covered
    # deduplicate, keep order
    seen, result = set(), []
    for s in starts:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Core tiled inference function
# ---------------------------------------------------------------------------
def run_model_tiled(model, rgb, device, save_np, tile_size=512, overlap=64):
    """
    Run TruFor inference using a sliding-window approach.

    For images that fit within (tile_size x tile_size) the model is called
    once directly.  For larger images the input is split into overlapping
    tiles; each tile is inferred independently and the results are stitched
    back to full resolution by weighted averaging.

    Parameters
    ----------
    model     : TruFor model (already in eval mode, on device)
    rgb       : torch.Tensor  (1, C, H, W) on device
    save_np   : bool
    tile_size : int  – maximum tile side length (default 512)
    overlap   : int  – overlap between neighbouring tiles (default 64)

    Returns   (all processed numpy arrays, ready for out_dict)
    -------
    pred_np  : np.ndarray (H, W) float32 – manipulation probability map
    conf_np  : np.ndarray (H, W) float32 or None
    det_val  : float or None              – image-level detection score
    npp_np   : np.ndarray (H, W) float32 or None
    """
    _, C, H, W = rgb.shape
    stride = max(1, tile_size - overlap)

    # ------------------------------------------------------------------
    # Fast path: image already fits in a single tile
    # ------------------------------------------------------------------
    if H <= tile_size and W <= tile_size:
        pred, conf, det, npp = model(rgb, save_np=save_np)

        pred = torch.squeeze(pred, 0)
        pred_np = F.softmax(pred, dim=0)[1].cpu().numpy()

        conf_np = None
        if conf is not None:
            conf = torch.squeeze(conf, 0)
            conf_np = torch.sigmoid(conf)[0].cpu().numpy()

        det_val = torch.sigmoid(det).item() if det is not None else None

        npp_np = None
        if npp is not None:
            npp_np = torch.squeeze(npp, 0)[0].cpu().numpy()

        return pred_np, conf_np, det_val, npp_np

    # ------------------------------------------------------------------
    # Tiled path
    # ------------------------------------------------------------------
    pred_acc  = np.zeros((H, W), dtype=np.float64)
    count_acc = np.zeros((H, W), dtype=np.float64)

    conf_acc  = None
    conf_cnt  = None
    npp_acc   = None
    npp_cnt   = None
    det_scores = []

    y_starts = _tile_starts(H, tile_size, stride)
    x_starts = _tile_starts(W, tile_size, stride)

    total_tiles = len(y_starts) * len(x_starts)
    print(f'  [Tiled] {H}x{W} image → {total_tiles} tiles '
          f'({len(y_starts)}r x {len(x_starts)}c), '
          f'tile={tile_size}, overlap={overlap}')

    for y_s in y_starts:
        y_e = y_s + min(tile_size, H - y_s)   # actual tile end (≤ H)
        for x_s in x_starts:
            x_e = x_s + min(tile_size, W - x_s)

            tile = rgb[:, :, y_s:y_e, x_s:x_e]   # (1, C, th, tw)

            try:
                pred_t, conf_t, det_t, npp_t = model(tile, save_np=save_np)
            except torch.cuda.OutOfMemoryError:
                # If even a single tile OOMs, fall back to a smaller sub-tile
                torch.cuda.empty_cache()
                th, tw = y_e - y_s, x_e - x_s
                scale  = 256 / max(th, tw)
                tile_s = F.interpolate(tile,
                                       size=(max(1, int(th * scale)),
                                             max(1, int(tw * scale))),
                                       mode='bilinear', align_corners=False)
                pred_t, conf_t, det_t, npp_t = model(tile_s, save_np=save_np)
                # upsample pred_t back to tile spatial size for stitching
                pred_t = F.interpolate(pred_t, size=(th, tw),
                                       mode='bilinear', align_corners=False)
                if conf_t is not None:
                    conf_t = F.interpolate(conf_t, size=(th, tw),
                                           mode='bilinear', align_corners=False)
                if npp_t is not None:
                    npp_t = F.interpolate(npp_t, size=(th, tw),
                                          mode='bilinear', align_corners=False)

            # --- process pred ---
            pred_t = torch.squeeze(pred_t, 0)
            pred_t_np = F.softmax(pred_t, dim=0)[1].cpu().numpy()   # (th, tw)
            th_out, tw_out = pred_t_np.shape

            pred_acc[y_s:y_s + th_out, x_s:x_s + tw_out] += pred_t_np
            count_acc[y_s:y_s + th_out, x_s:x_s + tw_out] += 1.0

            # --- process det ---
            if det_t is not None:
                det_scores.append(torch.sigmoid(det_t).item())

            # --- process conf ---
            if conf_t is not None:
                conf_t = torch.squeeze(conf_t, 0)
                conf_t_np = torch.sigmoid(conf_t)[0].cpu().numpy()
                if conf_acc is None:
                    conf_acc = np.zeros((H, W), dtype=np.float64)
                    conf_cnt = np.zeros((H, W), dtype=np.float64)
                conf_acc[y_s:y_s + th_out, x_s:x_s + tw_out] += conf_t_np
                conf_cnt[y_s:y_s + th_out, x_s:x_s + tw_out] += 1.0

            # --- process npp ---
            if npp_t is not None:
                npp_t = torch.squeeze(npp_t, 0)[0].cpu().numpy()
                if npp_acc is None:
                    npp_acc = np.zeros((H, W), dtype=np.float64)
                    npp_cnt = np.zeros((H, W), dtype=np.float64)
                npp_acc[y_s:y_s + th_out, x_s:x_s + tw_out] += npp_t
                npp_cnt[y_s:y_s + th_out, x_s:x_s + tw_out] += 1.0

    # ------------------------------------------------------------------
    # Average overlapping predictions
    # ------------------------------------------------------------------
    denom    = np.maximum(count_acc, 1e-8)
    pred_np  = (pred_acc / denom).astype(np.float32)

    conf_np  = (conf_acc / np.maximum(conf_cnt, 1e-8)).astype(np.float32) \
               if conf_acc is not None else None

    npp_np   = (npp_acc  / np.maximum(npp_cnt,  1e-8)).astype(np.float32) \
               if npp_acc  is not None else None

    det_val  = float(np.mean(det_scores)) if det_scores else None

    return pred_np, conf_np, det_val, npp_np


# ---------------------------------------------------------------------------
# Dataset / model setup
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
with torch.inference_mode():
    for index, (rgb, path) in enumerate(tqdm(testloader)):
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
        else:  # output is a filename
            filename_out = output

        if not filename_out.endswith('.npz'):
            filename_out = filename_out + '.npz'

        # Skip if a valid (complete) output already exists
        if os.path.isfile(filename_out):
            continue

        # Temporary file for atomic write (must end in .npz so numpy
        # does NOT auto-append .npz, which would break os.replace)
        tmp_out = filename_out[:-4] + '.tmp.npz'

        try:
            orig_imgsize = tuple(rgb.shape[2:])   # save before any tiling
            rgb = rgb.to(device)
            model.eval()

            # ---------------------------------------------------------
            # Tiled inference — returns processed numpy arrays directly
            # ---------------------------------------------------------
            pred, conf, det_val, npp = run_model_tiled(
                model, rgb, device, save_np,
                tile_size=TILE_SIZE, overlap=OVERLAP)

            out_dict = dict()
            out_dict['map'    ] = pred
            out_dict['imgsize'] = orig_imgsize
            if det_val is not None:
                out_dict['score'] = det_val
            if conf is not None:
                out_dict['conf'] = conf
            if save_np and npp is not None:
                out_dict['np++'] = npp

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
