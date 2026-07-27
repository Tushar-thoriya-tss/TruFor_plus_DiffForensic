# TruFor — Architecture & Training Workflow Reference

> Quick-lookup document so we don't re-read source files for every question.
> Diagrams referenced: [architecture.png](architecture.png), [phases.png](phases.png),
> [confidence.png](confidence.png), [noiseprint_pp.png](noiseprint_pp.png).

---

## 1. High-Level Overview

TruFor is an **image forgery localization + detection** model. Given an RGB image it produces:

| Output           | Shape (per image)       | Meaning                                                    |
|------------------|-------------------------|------------------------------------------------------------|
| **Anomaly map**  | `[2, H, W]` (logits)    | Per-pixel pristine vs. forged class scores                 |
| **Confidence map** | `[1, H, W]` (logit)   | Reliability of the anomaly map at every pixel (TCP target) |
| **Score (det)**  | scalar                  | Image-level forgery probability (after sigmoid)            |
| **Noiseprint++** | `[1 or 3, H, W]`        | Camera/editing fingerprint (intermediate)                  |

The whole network is referred to in code as `detconfcmx` (see [lib/utils.py:73](../TruFor_train_test/lib/utils.py#L73)).
The actual `nn.Module` is `EncoderDecoder` in
[lib/models/cmx/builder_np_conf.py:67](../TruFor_train_test/lib/models/cmx/builder_np_conf.py#L67).

```
        ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
RGB ───►├──► [DnCNN: Noiseprint++ extractor] ──► NP++ map ───┐        │
        │                                                    │        │
        │                                                    ▼        │
        │                        ┌──────────── Dual SegFormer ────────┤
        │                        │  RGB stream + NP++ stream          │
        │                        │  4 stages (1/4,1/8,1/16,1/32)      │
        │                        │  FRM (rectify) + FFM (fuse) per st │
        │                        └────┬───────────────────────────────┤
        │                             ▼                               │
        │                       fused multi-scale features            │
        │              ┌──────────────┼────────────────┐              │
        │              ▼              ▼                ▼              │
        │      MLP Anomaly Dec.  MLP Conf Dec.   (uses both ──►       │
        │       (2 ch logits)     (1 ch logit)    weighted pooling)   │
        │              │              │                ▼              │
        │              │              │       MLP Detector → score    │
        └──────────────┴──────────────┴───────────────────────────────┘
```

The model is **modular**: each module can be enabled (`MODULES`) and frozen (`FIX_MODULES`)
independently via config. This is what allows the 3-phase training strategy.

---

## 2. The Three Training Phases (`phases.png`)

| Phase | Goal                                  | Data                | Trains                        | Freezes              |
|-------|---------------------------------------|---------------------|-------------------------------|----------------------|
| **1** | Noiseprint++ extraction               | Pristine images only| DnCNN (NP++) — separate repo  | —                    |
| **2** | Anomaly localization (cross-modal)    | Pristine + forged   | Backbone + loc_head           | NP++                 |
| **3** | Confidence est. + Forgery detection   | Pristine + forged   | conf_head + det_head          | NP++, backbone, loc_head |

In this repo we **do not retrain Phase 1**. We get Phase-1 weights from
`pretrained_models/noiseprint++/noiseprint++.th`. Repo only does Phases 2 and 3.

Config files map to phases:
- Phase 2 (public TruFor datasets): [trufor_ph2.yaml](../TruFor_train_test/lib/config/trufor_ph2.yaml)
- Phase 2 (custom Aadhaar data):    [mydata_ph2.yaml](../TruFor_train_test/lib/config/mydata_ph2.yaml)
- Phase 3 (confidence + detector):  [trufor_ph3.yaml](../TruFor_train_test/lib/config/trufor_ph3.yaml)
  → loads `weights/mydata_ph2/best.pth.tar` via `TRAIN.PRETRAINING`.

---

## 3. Module-by-Module Architecture

### 3.1 Noiseprint++ Extractor — DnCNN
- File: [lib/models/DnCNN.py](../TruFor_train_test/lib/models/DnCNN.py)
- Built by `make_net(...)` with **17 conv layers**, 64 features, kernel 3, BN except first/last,
  ReLU except last (linear), no dilation.
- Input: 3-channel RGB in `[0, 1]` (already what the dataset returns: `t_RGB = … / 256.0`).
- Output channels: `NP_OUT_CHANNELS` (default 1). If 1, it's `torch.tile(..., (3,1,1))` so the
  encoder's NP++ branch receives 3 channels just like the RGB branch.
- Weights loaded from `cfg.MODEL.EXTRA.NP_WEIGHTS` (e.g. `pretrained_models/noiseprint++/noiseprint++.th`).
- **Always frozen** in our pipeline (`'NP++'` is always in `FIX_MODULES`).
- Purpose: produces a noise-residual "fingerprint" capturing editing history (resize, JPEG, etc.)
  — see [noiseprint_pp.png](noiseprint_pp.png).

### 3.2 Encoder — Dual SegFormer (CMX)
- File: [lib/models/cmx/encoders/dual_segformer.py](../TruFor_train_test/lib/models/cmx/encoders/dual_segformer.py)
- Default backbone: **MiT-B2** (`embed_dims=[64,128,320,512]`, `depths=[3,4,6,3]`).
- Two parallel SegFormer streams:
  - `block*`, `patch_embed*`, `norm*` ← RGB stream
  - `extra_block*`, `extra_patch_embed*`, `extra_norm*` ← Noiseprint++ stream
- After each of the **4 stages** the two streams are mixed by:
  1. **FRM — Feature Rectify Module** (channel + spatial attention) — see
     [net_utils.py:49](../TruFor_train_test/lib/models/cmx/net_utils.py#L49)
  2. **FFM — Feature Fusion Module** (cross-attention + channel embed) — see
     [net_utils.py:162](../TruFor_train_test/lib/models/cmx/net_utils.py#L162)
- Returns a list of **4 fused feature maps** at strides 4/8/16/32 → consumed by both decoders.
- ImageNet-pretrained backbone weights loaded from
  `pretrained_models/segformers/mit_b2.pth` (Phase 2 only — duplicated to both streams via
  `load_dualpath_model`).

### 3.3 Localization (Anomaly) Decoder — MLPDecoder
- File: [lib/models/cmx/decoders/MLPDecoder.py](../TruFor_train_test/lib/models/cmx/decoders/MLPDecoder.py)
- Standard SegFormer-style head: per-scale MLP → upsample to 1/4 → concat → 1×1 conv fuse → 1×1 conv predict.
- `embed_dim = 512` (set in YAML as `DECODER_EMBED_DIM`).
- **Output: 2-class logits** at 1/4 resolution → bilinearly upsampled to input size
  (`F.interpolate` in `encode_decode()`).
- Channel 0 = pristine score, Channel 1 = forged score. Softmax → probability map.

### 3.4 Confidence Decoder — MLPDecoder (1 channel)
- Same `DecoderHead` class as the localization decoder but with `num_classes=1`.
- Output: 1-channel logit per pixel, upsampled to image size.
- After `torch.sigmoid(...)` it estimates the **True Class Probability (TCP)** —
  see `criterion_conf.py` and [confidence.png](confidence.png):
  - TCP target = `pred_softmax[forged_class] * gt_forged + pred_softmax[pristine_class] * gt_pristine`
  - "1 if our prediction matched the truth, 0 if it didn't"
  - The conf head learns to **predict whether the localization is going to be right**.

### 3.5 Forgery Detection Head
- Defined in [builder_np_conf.py:120](../TruFor_train_test/lib/models/cmx/builder_np_conf.py#L120):

```python
nn.Sequential(
    nn.Linear(8, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 1),
)
```

- Input is **8 scalars** produced by `weighted_statistics_pooling` (see
  [layer_utils.py](../TruFor_train_test/lib/models/cmx/layer_utils.py)):
  - 4 from the confidence map (min, max, weighted-avg, weighted-mean-square)
  - 4 from `(loc[1] − loc[0])` weighted by `log_sigmoid(conf)`
- Output: single scalar logit → image-level forgery probability via sigmoid.
- Only used when `DETECTION: 'confpool'` (Phase 3).

---

## 4. The `forward` Path (one inference)

`EncoderDecoder.forward(rgb, save_np=False)`:

1. If `'NP++'` is in `MODS`, run DnCNN (under `no_grad` because it's always frozen) → `modal_x`.
   Tile to 3 channels if `NP_OUT_CHANNELS == 1`.
2. Apply RGB normalization based on `PREPRC`:
   - `imagenet` → standard ImageNet mean/std (this is what TruFor uses)
   - `xception` → `2x - 1`
   - `none`     → keep `[0, 1]`
3. Call `encode_decode(rgb, modal_x)`:
   - Backbone returns 4 fused feature maps.
   - Localization decoder → `out` (B×2×H×W after upsample).
   - Confidence decoder → `conf` (B×1×H×W after upsample), if enabled.
   - If detection enabled: pool `(conf, out)` → 8-feature vector → detector MLP → `det`.
4. Return `(out, conf, det, modal_x or None)`.

---

## 5. Training Workflow ([train.py](../TruFor_train_test/train.py))

### 5.1 Setup
- Parses `-exp` (config name without `.yaml`) and `-g` (GPU list).
- `update_config` merges `lib/config/<exp>.yaml` over the defaults in
  [lib/config/default.py](../TruFor_train_test/lib/config/default.py).
- Creates logger + TensorBoard writer in `weights/<exp>/` and `log/<model>/<exp>_<time>/`.
- Loads albumentations augmentation from `TRAIN.AUG` (default: `aug_res_comp.yaml`
  → `RandomScale ±0.5 p=0.5` + `ImageCompression jpeg 30–100 p=0.5`).

### 5.2 Datasets
- `myDataset` ([data_core.py](../TruFor_train_test/dataset/data_core.py)) wraps multiple
  sub-datasets selected by `DATASET.TRAIN / DATASET.VALID` keys:
  - `FR` → FantasticReality, `IMD` → IMD2020, `CA` → CASIA, `COCO` → tampCOCO,
    `RAISE` → compRAISE, `MyData` → custom Aadhaar (mostly tampered card images).
- **Class-balanced sampling at training time**: every epoch each sub-dataset contributes
  `smallest` samples (= size of the smallest sub-dataset, bounded by `TRAIN.NUM_SAMPLES > 0`).
  `len = smallest * num_sub_datasets`, and `__getitem__` round-robins by index.
- `train_dataset.shuffle()` shuffles each sub-dataset's `img_list` once per epoch.
- `AbstractDataset._create_tensor`:
  - Loads RGB → numpy → albumentations(image, mask) → optional pad to `crop_size` with
    `mask = -1` (ignore label) → random (grid-aligned if `grid_crop=True`) crop → optional
    centre crop to `max_dim` → `t_RGB = img/256.0`, `t_mask = long`.
- `MyData` ([dataset_MyData.py](../TruFor_train_test/dataset/dataset_MyData.py)):
  - Reads a comma-separated `image,mask` text list under
    `dataset_paths['MyData']` (set in [project_config.py](../TruFor_train_test/project_config.py)).
  - Mask binarized to `{0,1}` (`mask[mask>0]=1`). Transposes the mask if its shape is rotated.
- DataLoader:
  - Train: `batch_size = TRAIN.BATCH_SIZE_PER_GPU * len(gpus)`, shuffled.
  - Valid: `batch_size = 1` (arbitrary input sizes), unshuffled (to recover filenames).

### 5.3 Model + Loss Wrap
- `get_model(config)` → `EncoderDecoder` instance.
- Wrapped in `torch.nn.DataParallel`, then in `FullModel`
  ([lib/utils.py:40](../TruFor_train_test/lib/utils.py#L40)):
  - Stores per-loss criteria and weights from `LOSS.LOSSES`.
  - `forward(labels, rgbs)` runs the model, then per-loss compute, `final_loss = Σ w * loss`.
  - This pattern keeps loss computation **on each GPU's local replica**, reducing
    main-GPU memory.
- `get_optimizer`:
  - Uses `cmx/init_func.py:group_weight` to split params into BN, conv, bias groups (`cmx` rule).
  - Default optimizer is **SGD** (`momentum=0.9`, `wd=5e-4`, `nesterov=false`, `lr=5e-3` in our YAMLs).
  - Alternative: `adam`.

### 5.4 Loss Functions

`LOSS.LOSSES` is a list of `[name, weight, criterion]` triples
(see [lib/utils.py:80](../TruFor_train_test/lib/utils.py#L80)).

| Name   | Criterion                  | File                                     | When                          |
|--------|----------------------------|------------------------------------------|-------------------------------|
| `LOC`  | `cross_entropy` / `dice` / `binary_dice` / `dice_entropy` | [criterion.py](../TruFor_train_test/lib/core/criterion.py) | Phase 2 (default: `dice_entropy = 0.3 CE + 0.7 BinaryDice`) |
| `CONF` | `mse`                      | [criterion_conf.py](../TruFor_train_test/lib/core/criterion_conf.py) | Phase 3 |
| `DET`  | `cross_entropy` (weighted BCE) | [criterion_det.py](../TruFor_train_test/lib/core/criterion_det.py) | Phase 3 |

Notes:
- `CONF` is **MSE between sigmoid(conf) and TCP**, but only on the eroded-inside / dilated-outside
  region of the GT mask (computed by `calcolaGTs` with `erodeKernSize=15`, `dilateKernSize=11`).
  This excludes the noisy boundary.
- `DET` builds a binary target `target_det = (#non-ignored positive pixels > 3)` and uses a
  class-rebalancing weight `0.5/0.7` for positive vs `0.5/0.3` for negative.
- `IGNORE_LABEL = -1` (used everywhere — comes from padding in dataset).
- Phase-3 YAML default: `[['CONF', 1.0, 'mse'], ['DET', 0.5, 'cross_entropy']]`.

### 5.5 Learning Rate Schedule
- `adjust_learning_rate` (`lib/utils.py:33`): **polynomial decay**
  `lr = base_lr * (1 - cur_iters/max_iters)^0.9`.
- `num_iters = TRAIN.END_EPOCH * epoch_iters`. Called **every iter** inside `train()`.

### 5.6 Train Loop Skeleton

```python
for epoch in range(start_epoch, end_epoch):
    if epoch >= last_epoch:                # skip past-already-trained epochs on resume
        train_dataset.shuffle()
        train(epoch, ..., trainloader, optimizer, model, writer_dict, adjust_learning_rate)
        torch.cuda.empty_cache(); gc.collect()
        save → weights/<exp>/checkpoint.pth.tar

    if VALID.FIRST_VALID and start_epoch == last_epoch - 1: ...  # one validation before training
    metric_dict, IoU_array, conf_mat = validate(config, validloader, model, writer_dict, "valid")

    if metric_dict[best_key] better than best_value:
        save → weights/<exp>/best.pth.tar
```

`train()` ([lib/core/function.py:31](../TruFor_train_test/lib/core/function.py#L31)) just iterates
the loader, computes `losses.mean()`, backward, step, then logs `train_loss` and `learning_rate`.

`validate()` ([function.py:77](../TruFor_train_test/lib/core/function.py#L77)) is the heavy one —
see Metrics section below.

### 5.7 Checkpoints
- `checkpoint.pth.tar` written **every epoch** (epoch+1, best_value, best_key, state_dict, optimizer).
- `best.pth.tar` written whenever the chosen `VALID.BEST_KEY` improves
  (smaller for `*loss*`, larger otherwise).
- `TRAIN.RESUME=True` re-loads `checkpoint.pth.tar` from `final_output_dir` if present and
  enforces the **same `best_key`** (assertion). `last_epoch` = `checkpoint['epoch']`.
- `TRAIN.PRETRAINING` (non-empty) loads a **separate** initial weights file before resume —
  with a graceful retry that strips `detection.*` keys if shape-incompatible (Phase 2 → Phase 3).

---

## 6. Validation Metrics

Computed in [function.py:77+](../TruFor_train_test/lib/core/function.py#L77).

For every batch (size 1), upsample `pred` and `conf` to GT size, compute:

**Localization (PRED) metrics**
- Confusion matrix at pixel level (`get_confusion_matrix`).
- `avg_mse` — MSE between forged-class softmax and GT.
- `mIoU`, `mIoU_smooth` — mean IoU over the 2 classes (smoothed adds `+1` to num & den).
- `IoU_1_smooth` — IoU of forged class only.
- `p_mIoU`, `p_F1` — **permutation-invariant** versions (take `max(metric, swap-class metric)`).
  Useful when sign of forgery localization isn't deterministic.
- `pixel_acc` — global pixel accuracy after summing all confusion matrices.

**Confidence (CONF) metrics** — only when `conf_head` ∈ MODULES
- Confusion matrix using `tcp > 0.5` vs `conf > 0` (`get_confusion_matrix_1ch`).
- `avg_mse_CONF`, `avg_mIoU_CONF`, `avg_mIoU_smooth_CONF`.

**Detection (DET) metrics**
- `avg_det_tpr`, `avg_det_tnr` — running averages of (#correct positives / #positives) and
  the equivalent for negatives. `target_det = (count_nonzero(label*(label>=0)) > 3)`.
- `avg_det_bacc = (tpr + tnr) / 2` — used as `BEST_KEY` in Phase 3 YAML.

`avg_mse_CONF` etc. only added to `metric_dict` when `check_conf` is true.

---

## 7. Configuration Cheat-Sheet

Defaults live in [lib/config/default.py](../TruFor_train_test/lib/config/default.py).
Common per-experiment knobs in YAML:

| Key                                   | Meaning                                                            |
|---------------------------------------|--------------------------------------------------------------------|
| `MODEL.MODS`                          | Active input modalities, e.g. `('RGB','NP++')`                     |
| `MODEL.EXTRA.BACKBONE`                | `mit_b0..mit_b5` (default `mit_b2`)                                |
| `MODEL.EXTRA.DECODER`                 | Only `MLPDecoder` implemented                                      |
| `MODEL.EXTRA.DECODER_EMBED_DIM`       | Width of decoder MLP (TruFor uses 512)                             |
| `MODEL.EXTRA.PREPRC`                  | `imagenet` / `xception` / `none`                                   |
| `MODEL.EXTRA.NP_WEIGHTS`              | Path to pretrained Noiseprint++                                    |
| `MODEL.EXTRA.MODULES`                 | Subset of `['NP++','backbone','loc_head','conf_head','det_head']`  |
| `MODEL.EXTRA.FIX_MODULES`             | Same set, but frozen (`requires_grad=False`, eval-mode in forward) |
| `MODEL.EXTRA.DETECTION`               | `confpool` or `None`                                               |
| `LOSS.LOSSES`                         | List of `[name, weight, criterion]`                                |
| `DATASET.TRAIN` / `DATASET.VALID`     | Sub-dataset keys (FR/IMD/CA/COCO/RAISE/MyData)                     |
| `DATASET.CLASS_WEIGHTS`               | CE weights (`[bg, fg]`); our MyData uses `[0.1, 1.9]`              |
| `TRAIN.PRETRAINING`                   | Initial weights file (overrides BACKBONE pretrain at the load step)|
| `TRAIN.RESUME`                        | Resume from `weights/<exp>/checkpoint.pth.tar`                     |
| `TRAIN.AUG` / `VALID.AUG`             | Albumentations YAMLs (default `aug_res_comp.yaml`)                 |
| `TRAIN.BATCH_SIZE_PER_GPU`            | Per-GPU batch size                                                 |
| `TRAIN.LR / WD / MOMENTUM / NESTEROV` | Optimizer hyperparams                                              |
| `TRAIN.END_EPOCH`                     | Used for **LR scheduler horizon** even with EXTRA_EPOCH            |
| `TRAIN.EXTRA_EPOCH`                   | Run additional epochs past END_EPOCH (lr keeps decaying)           |
| `TRAIN.NUM_SAMPLES`                   | If > 0, cap per-epoch samples per sub-dataset                      |
| `VALID.MAX_SIZE`                      | Centre-crop validation images to this max side                     |
| `VALID.BEST_KEY`                      | Metric used to decide `best.pth.tar`                               |
| `VALID.FIRST_VALID`                   | If True, run one validation before training starts                 |

`MODEL.PRETRAINED` (separate from `TRAIN.PRETRAINING`) → ImageNet-pretrained SegFormer weights
for the **backbone only**, applied in `EncoderDecoder.init_weights`.

---

## 8. Inference ([test.py](../TruFor_train_test/test.py))

### 8.1 CLI

```bash
python test.py \
    -g 0                            # GPU id, -1 = CPU
    -in  ../images                  # single file, directory, OR glob pattern (e.g. 'imgs/**/*.jpg')
    -out ../output                  # output dir or single .npz filename
    -exp trufor_ph3                 # YAML name (without .yaml) — picks up MODEL/MODS/MODULES + TEST.MODEL_FILE
    -save_np                        # optional flag — also dump the raw Noiseprint++ map
    [opts ...]                      # extra YACS overrides, e.g. TEST.MODEL_FILE ../weights/foo.pth.tar
```

The checkpoint path is taken from `config.TEST.MODEL_FILE` (set in the YAML
or via `update_config` fallback to `weights/trufor.pth.tar`). For our setup
Phase-3 YAML overrides it to `../weights/custom_data_train_eph_100.pth.tar`.

### 8.2 What test.py does, step by step

1. **Argparse + `update_config(config, args)`** — same config loader as training, but
   `MODEL.PRETRAINED` / `TRAIN.PRETRAINING` are irrelevant here.
2. **Device** — `cuda:<gpu>` if `gpu >= 0`, else `cpu`. cudnn flags applied for GPU runs.
3. **Build `list_img`**:
   - `'*' in input` → `glob(input, recursive=True)` (skip directories).
   - File → single-item list.
   - Directory → `glob(input + '/**/*', recursive=True)` (skip directories).
4. **Image verification pass** (lines 72–89) — for every candidate:
   - `Image.open(...).verify()` (structural check) then a second open + `.convert("RGB")`
     to force a full decode.
   - Skips silently on any exception. Logs the **count** of valid images.
   - This is slow but cheap insurance against corrupt files in the input glob.
5. **`TestDataset(list_img=list_img)`** ([dataset/dataset_test.py](../TruFor_train_test/dataset/dataset_test.py)):
   - `__getitem__` returns `(torch.tensor(img_RGB.transpose(2,0,1)) / 256.0, rgb_path)`.
   - **No augmentation, no crop, no resize.** Images keep their native resolution.
6. **DataLoader** — `batch_size=1` (mandatory because every image has its own size; the
   model handles arbitrary H, W).
7. **Load model**:
   - `get_model(config)` → `EncoderDecoder` (NOT wrapped in `FullModel` / `DataParallel`
     during inference).
   - `torch.load(..., weights_only=False)` → reads `epoch` and `state_dict`.
   - `model.load_state_dict(checkpoint['state_dict'])` — works because the saved
     state dict was already `model.model.module.state_dict()` (module-level) at train time.
   - `model.to(device)`.
8. **Inference loop** under `with torch.inference_mode():`:
   1. **Output filename** logic (line 114-131):
      - If `output` has no extension → treat it as a directory and mirror the input's
        sub-path under it: `os.path.join(output, sub_path) + '.npz'`.
        - `sub_path` is computed by stripping the **root** part of `input` (the part
          before any `*` glob), so the directory structure under `input` is preserved.
      - Else `output` is taken as a single filename.
      - Ensures the filename ends in `.npz`.
   2. **Skip-if-exists**: `if not os.path.isfile(filename_out): ...` — running twice will
      not overwrite. To force re-inference, delete the `.npz`.
   3. `model.eval()` → forward `pred, conf, det, npp = model(rgb, save_np=save_np)`:
      - `pred` → `[1, 2, H, W]` logits at full resolution.
      - `conf` → `[1, 1, H, W]` logits (None if no `conf_head`).
      - `det`  → `[1, 1]` logit (None if no `det_head`).
      - `npp`  → raw NP++ output (only when `save_np=True`).
   4. **Post-process**:
      - `pred  = softmax(pred[0], dim=0)[1].cpu().numpy()`  → **forged-class probability**, shape `(H, W)`.
      - `conf  = sigmoid(conf[0])[0].cpu().numpy()`         → confidence in `[0, 1]`, shape `(H, W)`.
      - `det_sig = sigmoid(det).item()`                     → scalar in `[0, 1]`.
      - `npp   = npp[0][0].cpu().numpy()`                   → first NP++ channel, shape `(H, W)`.
   5. **Write `.npz`**:
      - `out_dict['map']     = pred`             *(always)*
      - `out_dict['imgsize'] = tuple(rgb.shape[2:])` *(H, W) of the tensor that was fed in*
      - `out_dict['score']   = det_sig`          *(only if det head present)*
      - `out_dict['conf']    = conf`             *(only if conf head present)*
      - `out_dict['np++']    = npp`              *(only if `--save_np` was passed)*
      - `makedirs(dirname, exist_ok=True)` then `np.savez(filename_out, **out_dict)`.
   6. The whole per-image body is wrapped in `try/except` with `traceback.print_exc()` —
      a single bad image will not stop the run.

### 8.3 Reading the outputs

```python
import numpy as np
data = np.load('output/example.jpg.npz')
fmap  = data['map']      # (H, W) float32 — forged-class probability
size  = data['imgsize']  # (H, W) ints — useful if you resized for display
score = float(data['score'])  if 'score' in data.files else None
conf  = data['conf']     if 'conf'  in data.files else None
npp   = data['np++']     if 'np++'  in data.files else None
```

Typical visualization:
- Localization: `fmap` as a heatmap overlaid on the input.
- Confidence: `conf` as a separate heatmap (high = trust the localization).
- Detection: threshold `score > 0.5` for binary forged/clean.

### 8.4 Inference variants in the repo

| Script                                  | Use case                                                                 |
|-----------------------------------------|--------------------------------------------------------------------------|
| `test.py`                               | Default: native-resolution single-pass inference.                       |
| `tiled_test.py`                         | Sliding-window inference for **very large** images that don't fit in VRAM.|
| `progressive_downsc_test.py`            | Run inference at progressively downscaled sizes and aggregate.          |
| `progressive_downsc_test_2.py`          | Variant of the above (tweaked aggregation / stopping rule).             |
| `Test_TruFor.py` (repo root)            | Standalone wrapper around the same logic, used for ad-hoc testing.      |
| `export_trufor_to_onnx.py` (repo root)  | Exports the trained model to ONNX for deployment outside PyTorch.       |

### 8.5 Inference gotchas

- **Input scaling is `/256.0`**, NOT `/255.0`. Match this exactly when running the model
  outside `TestDataset` (e.g. in a notebook). The Noiseprint++ network was trained with
  this convention.
- **`config.TEST.MODEL_FILE` is the only checkpoint hook at inference.** Setting
  `TRAIN.PRETRAINING` / `MODEL.PRETRAINED` has no effect during `test.py`.
- **Skip-if-exists is silent.** When iterating on the model, re-running `test.py` on the
  same output dir is a no-op until you clear the `.npz` files.
- **State-dict path matches the train-time save**: training saves `model.model.module.state_dict()`
  and inference does `get_model(...).load_state_dict(...)`. Don't wrap the inference model
  in `FullModel` or `DataParallel` — keys won't match.
- **Image verification pass is O(N) decodes** before inference even starts. For huge
  globs, comment lines 72–89 if you trust your input.
- **Memory at native resolution**: very large RGB inputs run the full SegFormer twice
  (RGB + NP++ stream). On consumer GPUs prefer `tiled_test.py` for 4K+ images.
- **`save_np` adds NP++ to the output but also forces `dncnn` to keep its tensor across
  the forward**; harmless but slightly more VRAM.

---

## 9. Phase-by-Phase Recipe (Practical)

### Phase 2 — Localization
```bash
# our custom data
python train.py -exp mydata_ph2 -g 0

# config flags that matter:
# MODULES:    ['NP++','backbone','loc_head']
# FIX_MODULES:['NP++']
# LOSS:       [['LOC', 1.0, 'dice_entropy']]
# BEST_KEY:   'avg_p-F1_smooth'
# PRETRAINED: 'pretrained_models/segformers/mit_b2.pth'  # backbone init
# NP_WEIGHTS: 'pretrained_models/noiseprint++/noiseprint++.th'
```

Output: `weights/mydata_ph2/{checkpoint,best}.pth.tar`.

### Phase 3 — Confidence + Detection
```bash
python train.py -exp trufor_ph3 -g 0

# config flags that matter:
# MODULES:    ['NP++','backbone','loc_head','conf_head','det_head']
# FIX_MODULES:['NP++','backbone','loc_head']               # freeze phase 2
# LOSS:       [['CONF', 1.0, 'mse'], ['DET', 0.5, 'cross_entropy']]
# DETECTION:  'confpool'
# BEST_KEY:   'avg_det_bacc'
# PRETRAINING: 'weights/mydata_ph2/best.pth.tar'           # load phase 2 weights
```

The try/except in `train.py` (around line 129) handles the case where the Phase-2 file has no
`detection.*` keys: it loads with `strict=False` and silently drops the detection branch keys.

---

## 10. Quick File Map

| Topic                         | File                                                                                |
|------------------------------|--------------------------------------------------------------------------------------|
| Entry — training              | [TruFor_train_test/train.py](../TruFor_train_test/train.py)                          |
| Entry — inference             | [TruFor_train_test/test.py](../TruFor_train_test/test.py)                            |
| Whole model assembly          | [lib/models/cmx/builder_np_conf.py](../TruFor_train_test/lib/models/cmx/builder_np_conf.py) |
| Noiseprint++ (DnCNN)          | [lib/models/DnCNN.py](../TruFor_train_test/lib/models/DnCNN.py)                      |
| Encoder (dual SegFormer)      | [lib/models/cmx/encoders/dual_segformer.py](../TruFor_train_test/lib/models/cmx/encoders/dual_segformer.py) |
| FRM + FFM fusion              | [lib/models/cmx/net_utils.py](../TruFor_train_test/lib/models/cmx/net_utils.py)      |
| MLPDecoder (loc + conf head)  | [lib/models/cmx/decoders/MLPDecoder.py](../TruFor_train_test/lib/models/cmx/decoders/MLPDecoder.py) |
| Detector pooling op           | [lib/models/cmx/layer_utils.py](../TruFor_train_test/lib/models/cmx/layer_utils.py)  |
| FullModel wrapper + helpers   | [lib/utils.py](../TruFor_train_test/lib/utils.py)                                    |
| Train/Validate loops + metrics| [lib/core/function.py](../TruFor_train_test/lib/core/function.py)                    |
| Losses — localization         | [lib/core/criterion.py](../TruFor_train_test/lib/core/criterion.py)                  |
| Loss — confidence (MSE on TCP)| [lib/core/criterion_conf.py](../TruFor_train_test/lib/core/criterion_conf.py)        |
| Loss — detection (weighted BCE)| [lib/core/criterion_det.py](../TruFor_train_test/lib/core/criterion_det.py)         |
| Config defaults               | [lib/config/default.py](../TruFor_train_test/lib/config/default.py)                  |
| Aug pipeline                  | [lib/config/aug_res_comp.yaml](../TruFor_train_test/lib/config/aug_res_comp.yaml)    |
| Dataset router                | [dataset/data_core.py](../TruFor_train_test/dataset/data_core.py)                    |
| Base dataset (crop/pad/aug)   | [dataset/AbstractDataset.py](../TruFor_train_test/dataset/AbstractDataset.py)        |
| Custom dataset                | [dataset/dataset_MyData.py](../TruFor_train_test/dataset/dataset_MyData.py)          |
| Inference dataset             | [dataset/dataset_test.py](../TruFor_train_test/dataset/dataset_test.py)              |
| Dataset paths                 | [project_config.py](../TruFor_train_test/project_config.py)                          |

---

## 11. Things That Are Easy To Get Wrong

- **Input range**: `dataset` returns `RGB / 256.0` (not `/255.0`). The Noiseprint++ network was
  trained with this range — don't "fix" the division.
- **NP++ is always frozen**, even in Phase 2/3. It also runs in `eval()` inside a `no_grad` block.
- **Backbone pretraining vs full pretraining are separate**:
  - `MODEL.PRETRAINED` = SegFormer backbone-only weights (applied in `init_weights`).
  - `TRAIN.PRETRAINING` = full checkpoint of an earlier phase (applied in `train.py` after `get_model`).
- **DataParallel + FullModel**: `model.model.module.load_state_dict(...)` is the correct path
  (model → FullModel → DataParallel → real module). Don't drop the `.module`.
- **`best_key` is asserted on resume**. Changing `BEST_KEY` between Phase 2 and Phase 3 is fine
  because Phase 3 starts from `TRAIN.PRETRAINING` (which doesn't trigger the assert), not from
  `RESUME` of a Phase-2 checkpoint.
- **Class-balanced sampling** caps each sub-dataset to the size of the smallest one per epoch —
  if you add a tiny sub-dataset, the others will be heavily under-sampled.
- **Validation batch size must be 1** because images keep arbitrary sizes (the filename order is
  also relied upon — `shuffle=False`).
- **`ignore_label = -1`** comes from `_create_tensor` padding. It must be excluded from CE and
  Dice losses and from confusion matrices.
