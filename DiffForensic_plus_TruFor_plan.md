# DiffForensic + TruFor — Implementation Plan (ECEM Edge-Cue Branch)

> **Audience:** the coding agent (Claude Opus) that will implement this in an isolated copy
> of the TruFor repo. This plan is self-contained: every integration point below was
> verified against the actual source files, with line references. Follow the steps in
> order. Do not redesign — where a choice was possible, the decision is already made and
> the reason is stated.
>
> **Background reading (optional):** [docs/DiffForensic_plus_TruFor.md](docs/DiffForensic_plus_TruFor.md)
> — the analysis that led to this plan. Paper scans in `DiffForensics paper/`.
> Architecture reference: [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md).

---

## 0. Goal (one paragraph)

Add DiffForensics' **Edge Cue Enhancement Module (ECEM)** and its **multi-scale weighted
edge dice loss (`L_edg`)** to TruFor as an *auxiliary training branch* on the localization
decoder. This is supervision-only: inference outputs, `test.py`, and the ONNX export remain
byte-identical unless a new flag is passed. Everything else in DiffForensics (DDPM/Simplex
pre-training, frozen-ADE20K encoder, their detection head) is **explicitly out of scope** —
see the analysis doc for why.

**Scope note:** Sections 1–6 are the *core* integration (Track A) and must land first.
Section 7 is an ordered roadmap of further DiffForensics-derived experiments (Tracks B–F)
to run after the core gate passes — work through them in the given order, one at a time,
logging results as described in §7.7.

Formulas being implemented (DiffForensics Eq. 2, 3, 7):

```
g_k   = | V ∗ | H ∗ d_k | |            H = [1,−1] row-diff, V = [1,−1]ᵀ col-diff, |·| = abs
f_k^e = Upsample( σ( Conv3×3( d_k − g_k ) ) )
L_edg = Σ_{k=1..3}  (1 / 2^(k−1)) · dice( f_k^e , y^e )      k=1 finest scale, weight 1.0
```

`y^e` = boundary band of the GT tamper mask (morphological gradient), built on the fly.

---

## 1. Ground Rules (invariants — do not violate)

1. **Never modify** the Noiseprint++ path: `lib/models/DnCNN.py`, the `/256.0` input
   scaling, or the `'NP++'`-always-frozen convention.
2. **`ignore_label = -1`** (from dataset padding) must be excluded from every new loss
   term, the same way `BinaryDiceLoss` does it with `valid_mask`
   ([criterion.py:95-134](TruFor_train_test/lib/core/criterion.py#L95-L134)).
3. **Backward compatibility of forward signatures.** `test.py:` unpacks
   `pred, conf, det, npp = model(rgb, save_np=save_np)` and
   `validate()` unpacks `losses, pred, conf, det = model(labels=..., rgbs=...)`.
   Both must keep working unchanged. New outputs only appear behind a new
   `return_edges=True` kwarg / an `'EDG'` loss entry.
4. **DataParallel constraint:** edge maps must be *returned* from
   `EncoderDecoder.forward` (DataParallel gathers only return values — never stash
   tensors on `self`, replica attributes are discarded).
5. `DecoderHead` is instantiated twice (loc head AND conf head,
   [builder_np_conf.py:104-116](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L104-L116)).
   Any change to it must use default-valued kwargs so the conf head is unaffected.
6. All commands run from `TruFor_train_test/`. Training entry:
   `python train.py -exp <yaml-name> -g 0`.
7. Work in the isolated copy only. Keep a git commit per step so any regression bisects
   trivially.

---

## 2. Architecture After the Change

```
                     fused encoder features  x = [c1,c2,c3,c4]  (strides 4/8/16/32)
                                   │
                     ┌─────────────┴──────────────┐
                     ▼                            ▼
              DecoderHead (loc)             DecoderHead (conf)     ← untouched
              per-scale MLP proj:
        _c1 (512ch, 1/4)  ──────────────► ECEM_1 ─► e1 ─► up ─┐
        _c2 (512ch, 1/8, native) ───────► ECEM_2 ─► e2 ─► up ─┼─► L_edg (train only)
        _c3 (512ch, 1/16, native) ──────► ECEM_3 ─► e3 ─► up ─┘
        _c4..._c1 upsampled+concat → fuse → 2-ch logits  → L_loc  ← untouched
```

Key design decisions (already made):
- **Tap points:** the per-scale MLP-projected features `_c1/_c2/_c3` at their **native**
  resolutions (strides 4/8/16), *before* the upsample-to-1/4 that
  [MLPDecoder.py:66-75](TruFor_train_test/lib/models/cmx/decoders/MLPDecoder.py#L66-L75)
  applies. This preserves the paper's coarse→fine multi-scale character. `_c4` (stride 32)
  is skipped — the paper also uses only 3 scales.
- All three tapped features have `DECODER_EMBED_DIM = 512` channels (post-MLP), so each
  ECEM is `Conv2d(512, 1, 3, padding=1)` — tiny (≈4.6k params each).
- ECEM outputs **logits**; sigmoid is applied inside the loss (numerically standard).
- Edge GT is generated **inside the loss** from the label tensor via GPU max-pooling —
  zero dataset/dataloader changes, automatically correct under crops/augs/padding.
- New module name: `'edge_head'` (joins `['NP++','backbone','loc_head','conf_head','det_head']`).
- New loss name: `'EDG'`, criterion string `'edge_dice'`.

---

## 3. Implementation Steps

### Step 1 — New file: `TruFor_train_test/lib/models/cmx/ecem.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_enhance(d):
    """g = |V * |H * d||  (Eq. 2 of DiffForensics).
    Adjacent-row abs difference, then adjacent-column abs difference.
    Zero-pad the last row/col so the output keeps the input's spatial size."""
    h = torch.abs(d[:, :, 1:, :] - d[:, :, :-1, :])
    h = F.pad(h, (0, 0, 0, 1))
    g = torch.abs(h[:, :, :, 1:] - h[:, :, :, :-1])
    g = F.pad(g, (0, 1, 0, 0))
    return g


class ECEM(nn.Module):
    """Edge Cue Enhancement Module (DiffForensics, Fig. 4).
    Consumes a decoder feature map d_k, returns a 1-channel edge logit map
    at the same spatial size (upsampling to image size happens in the caller;
    sigmoid happens in the loss)."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, d):
        g = edge_enhance(d)
        return self.conv(d - g)
```

### Step 2 — `TruFor_train_test/lib/models/cmx/decoders/MLPDecoder.py`

Modify `DecoderHead.forward` (currently
[lines 59-85](TruFor_train_test/lib/models/cmx/decoders/MLPDecoder.py#L59-L85)) to
optionally return the native-resolution per-scale features. Keep the existing
`return_feats` behavior intact. Replace the forward with:

```python
    def forward(self, inputs, return_feats=False, return_scale_feats=False):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4_up = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3_up = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2_up = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1)
        x = self.linear_fuse(_c)
        x = self.dropout(x)
        x = self.linear_pred(x)

        if return_scale_feats:
            # fine -> coarse: strides 4, 8, 16 (all embed_dim channels, native res)
            return x, (_c1, _c2, _c3)
        if return_feats:
            return x, _c
        return x
```

(Only renames `_c4/_c3/_c2` interpolated tensors to `*_up` and adds the new branch — the
computation is numerically identical.)

### Step 3 — `TruFor_train_test/lib/models/cmx/builder_np_conf.py`

**3a. Register the module name.** At
[line 81](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L81):

```python
        modules_list = ['NP++','backbone','loc_head','conf_head','det_head','edge_head']
```

**3b. Build the edge heads.** Inside the `if self.cfg.DECODER == 'MLPDecoder':` block,
after the detection-head construction (after
[line 130](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L120-L130)):

```python
            # edge cue enhancement heads (DiffForensics ECEM) — auxiliary, train-time only
            self.edge_heads = None
            if 'edge_head' in self.cfg.MODULES:
                assert 'loc_head' in self.cfg.MODULES
                from .ecem import ECEM
                self.edge_heads = nn.ModuleList(
                    [ECEM(self.cfg.DECODER_EMBED_DIM) for _ in range(3)])
```

(Also add `self.edge_heads = None` next to the other head declarations around line 93-95
so the attribute always exists.)

**3c. Init + freeze support.** In `init_weights`
([lines 176-204](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L176-L204)), after the
conf-head init block add:

```python
        if self.edge_heads:
            init_weight(self.edge_heads, nn.init.kaiming_normal_,
                        self.norm_layer, self.cfg.BN_EPS, self.cfg.BN_MOMENTUM,
                        mode='fan_in', nonlinearity='relu')
```

and at the end of the freezing section:

```python
        if 'edge_head' in self.cfg.FIX_MODULES:
            for param in self.edge_heads.parameters():
                param.requires_grad = False
```

**3d. `encode_decode` — compute edges.** Change the signature to
`def encode_decode(self, rgb, modal_x, return_edges=False):` and replace the
anomaly-localization block ([lines 224-232](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L224-L232)) with:

```python
        # anomaly localization
        need_edges = return_edges and (self.edge_heads is not None)
        if 'loc_head' in self.cfg.FIX_MODULES:
            with torch.no_grad():
                self.decode_head.eval()
                out = self.decode_head(x, return_scale_feats=need_edges)
        else:
            out = self.decode_head(x, return_scale_feats=need_edges)

        if need_edges:
            out, scale_feats = out
            edges = []
            for head, feat in zip(self.edge_heads, scale_feats):
                e = head(feat)
                e = F.interpolate(e, size=orisize[2:], mode='bilinear', align_corners=False)
                edges.append(e)
        else:
            edges = None

        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
```

and change the return ([line 260](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L260)) to
`return out, conf, det, edges`.

**3e. `forward` — thread it through.** Change the signature to
`def forward(self, rgb, save_np=False, return_edges=False):` and the tail
([lines 290-296](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L290-L296)) to:

```python
        # Localization and Detection
        out, conf, det, edges = self.encode_decode(rgb, modal_x, return_edges=return_edges)

        npp = modal_x if save_np else None
        if return_edges:
            return out, conf, det, npp, edges
        return out, conf, det, npp
```

Default call sites (`test.py`, ONNX export, FullModel without EDG) still get exactly the
old 4-tuple.

### Step 4 — `TruFor_train_test/lib/core/criterion.py` — new `EdgeDiceLoss`

Append to the file:

```python
class EdgeDiceLoss(nn.Module):
    """Multi-scale edge supervision (DiffForensics Eq. 7).
    Edge GT is the boundary band of the tamper mask: dilate(m) - erode(m),
    computed on the fly with max-pooling. Pixels within `radius` of an
    ignore-label pixel are excluded, so padding borders never produce
    spurious edge supervision.
    smooth=1.0 is intentional and must NOT inherit config.LOSS.SMOOTH=0:
    with an all-background edge GT (pristine image) it makes the dice loss
    ~0 instead of ~1."""

    def __init__(self, ignore_label=-1, radius=2, scale_weights=(1.0, 0.5, 0.25),
                 smooth=1.0, exponent=2):
        super(EdgeDiceLoss, self).__init__()
        self.ignore_label = ignore_label
        self.radius = radius
        self.scale_weights = scale_weights
        self.smooth = smooth
        self.exponent = exponent

    def _mask_to_edge_band(self, target):
        # target: B x H x W long in {-1, 0, 1}
        k = 2 * self.radius + 1
        m = (target == 1).float().unsqueeze(1)                     # forged
        dil = F.max_pool2d(m, k, stride=1, padding=self.radius)
        ero = 1.0 - F.max_pool2d(1.0 - m, k, stride=1, padding=self.radius)
        edge = (dil - ero).clamp(0, 1).squeeze(1)                  # B x H x W
        inv = (target == self.ignore_label).float().unsqueeze(1)
        inv = F.max_pool2d(inv, k, stride=1, padding=self.radius)  # grow ignore zone
        valid = (1.0 - inv).squeeze(1)
        return edge, valid

    def _binary_dice(self, pred, target, valid_mask):
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent) * valid_mask
                        + target.pow(self.exponent) * valid_mask, dim=1) \
              + max(self.smooth, 1e-5)
        return 1 - torch.mean(num / den)

    def forward(self, edge_logits_list, target):
        edge_gt, valid = self._mask_to_edge_band(target)
        loss = 0.0
        for w, e in zip(self.scale_weights, edge_logits_list):
            if e.shape[-2:] != target.shape[-2:]:
                e = F.interpolate(e, size=target.shape[-2:], mode='bilinear')
            prob = torch.sigmoid(e).squeeze(1)                     # B x H x W
            loss = loss + w * self._binary_dice(prob, edge_gt, valid)
        return loss
```

### Step 5 — `TruFor_train_test/lib/utils.py` — wire the loss

**5a. `get_criterion`** ([lines 80-127](TruFor_train_test/lib/utils.py#L80-L127)):
- change init to `criterion_loc, criterion_conf, criterion_det, criterion_edge = None, None, None, None`
- extend the name assertion: `assert l in ['LOC', 'CONF', 'DET', 'EDG']`
- add a branch:

```python
        # Training the Edge branch (DiffForensics ECEM)
        elif l == 'EDG':
            if criterion == 'edge_dice':
                from lib.core.criterion import EdgeDiceLoss
                criterion_edge = EdgeDiceLoss(
                    ignore_label=ignore_label,
                    radius=config.LOSS.EDGE_RADIUS,
                    scale_weights=tuple(config.LOSS.EDGE_SCALE_WEIGHTS)).cuda()
            else:
                raise ValueError('Edge loss not implemented')
```
- return the 4-tuple: `return criterion_loc, criterion_conf, criterion_det, criterion_edge`

**5b. `FullModel`** ([lines 40-66](TruFor_train_test/lib/utils.py#L40-L66)):

```python
        self.loss_loc, self.loss_conf, self.loss_det, self.loss_edge = get_criterion(config)
        self.use_edges = any(l == 'EDG' for (l, w, c) in self.losses)

    def forward(self, labels=None, rgbs=None):
        if self.use_edges:
            outputs, conf, det, npp, edges = self.model(rgbs, return_edges=True)
        else:
            outputs, conf, det, npp = self.model(rgbs)
            edges = None
        final_loss = 0
        for (l,w,_) in self.losses:
            if l == 'LOC':
                loss = self.loss_loc(outputs, labels)
            elif l == 'CONF':
                loss = self.loss_conf(outputs, labels, conf)
            elif l == 'DET':
                loss = self.loss_det(det, labels)
            elif l == 'EDG':
                loss = self.loss_edge(edges, labels)

            loss = torch.unsqueeze(loss, 0)
            final_loss += w * loss

        return final_loss, outputs, conf, det
```

Note the return stays a 4-tuple — `validate()` in
[function.py:114](TruFor_train_test/lib/core/function.py#L114) keeps working, and the EDG
term is automatically included in train/valid `avg_loss`.

### Step 6 — `TruFor_train_test/lib/config/default.py` — new knobs

`_C.LOSS` is **not** `new_allowed`, so the defaults must be added there (after
[line 60](TruFor_train_test/lib/config/default.py#L60)):

```python
_C.LOSS.EDGE_RADIUS = 2                      # half-width (px) of the GT boundary band
_C.LOSS.EDGE_SCALE_WEIGHTS = [1.0, 0.5, 0.25]  # fine (1/4), mid (1/8), coarse (1/16)
```

(`MODEL.EXTRA` is `new_allowed=True`, so `MODULES: [... , 'edge_head']` needs no default change.)

### Step 7 — New config: `TruFor_train_test/lib/config/mydata_ph2_edge.yaml`

Copy of `mydata_ph2.yaml` with the three highlighted changes:

```yaml
CUDNN:
  BENCHMARK: false
  DETERMINISTIC: false
  ENABLED: false
WORKERS: 24

DATASET:
  TRAIN: [MyData]
  VALID: [MyData]
  NUM_CLASSES: 2
  CLASS_WEIGHTS: [0.1, 1.9]  # for my aadhar tamper data imbalance
MODEL:
  NAME: detconfcmx
  PRETRAINED: 'pretrained_models/segformers/mit_b2.pth'
  MODS: ('RGB','NP++')
  EXTRA:
      BACKBONE: mit_b2
      DECODER: MLPDecoder
      DECODER_EMBED_DIM: 512
      PREPRC: 'imagenet'
      BN_EPS: 0.001
      BN_MOMENTUM: 0.1
      NP_WEIGHTS: 'pretrained_models/noiseprint++/noiseprint++.th'
      MODULES: ['NP++','backbone','loc_head','edge_head']        # CHANGED
      FIX_MODULES: ['NP++']
LOSS:
  LOSSES:
    - [ 'LOC', 1.0, 'dice_entropy' ]
    - [ 'EDG', 0.8, 'edge_dice' ]                                # NEW
  SMOOTH: 0
  EDGE_RADIUS: 2                                                 # NEW
  EDGE_SCALE_WEIGHTS: [1.0, 0.5, 0.25]                           # NEW
TRAIN:
  PRETRAINING: 'weights/mydata_ph2/best.pth.tar'                 # warm start (see note)
  IMAGE_SIZE: [512,512]
  BATCH_SIZE_PER_GPU: 4
  SHUFFLE: true
  BEGIN_EPOCH: 0
  END_EPOCH: 100
  OPTIMIZER: sgd
  LR: 0.005
  WD: 0.0005
  MOMENTUM: 0.9
  NESTEROV: false
  IGNORE_LABEL: -1
  NUM_SAMPLES: -1
  AUG: 'lib/config/aug_res_comp.yaml'
VALID:
  FIRST_VALID: false
  MAX_SIZE: 512
  BEST_KEY: 'avg_p-F1_smooth'
  AUG: 'lib/config/aug_res_comp.yaml'
```

**Warm-start note:** `train.py` already loads `TRAIN.PRETRAINING` with `strict=False`
([train.py:129-133](TruFor_train_test/train.py#L129-L133)), so the missing
`edge_heads.*` keys are tolerated — no loader change needed. If the warm start turns out
to converge to the old solution too eagerly (edge branch under-trained), fall back to
`PRETRAINING:` empty (train Phase 2 from ImageNet+NP++ as usual). Warm start first; it's
much cheaper. Consider `LR: 0.002` and `END_EPOCH: 50` for the warm-started run.

### Step 8 — New config: `TruFor_train_test/lib/config/trufor_ph3_edge.yaml`

Copy of `trufor_ph3.yaml` with:

```yaml
      MODULES: ['NP++','backbone','loc_head','edge_head','conf_head','det_head']
      FIX_MODULES: ['NP++','backbone','loc_head','edge_head']   # edge branch frozen in ph3
```
```yaml
TRAIN:
  PRETRAINING: 'weights/mydata_ph2_edge/best.pth.tar'
```
and its own `TEST.MODEL_FILE` pointing at the eventual ph3-edge best checkpoint.
`LOSS.LOSSES` stays `[['CONF',1.0,'mse'], ['DET',0.5,'cross_entropy']]` — **no `'EDG'`
in Phase 3** (loc_head is frozen there; its features run under `no_grad`, and the edge
branch's job is done). `edge_head` must stay in `MODULES` so the checkpoint's
`edge_heads.*` keys load cleanly.

### Step 9 — Verification tooling (new file `TruFor_train_test/tools_edge_debug.py`)

A small standalone script the implementer must write and run, which:
1. Builds `EdgeDiceLoss` and feeds it a synthetic 1×64×64 mask (a filled square, with a
   `-1` padded border) — asserts: edge band is a hollow rectangle, band pixels within
   `radius` of the padding are excluded, pristine mask (all zeros) yields loss `< 0.05`,
   perfect prediction yields loss `< 0.05`, inverted prediction yields loss `> 0.9`.
2. Loads 3 real samples via `myDataset` with the `mydata_ph2_edge` config and saves
   side-by-side PNGs (`image | mask | edge_gt`) to `debug_edge_gt/` for eyeballing.
3. Instantiates the full model from `mydata_ph2_edge.yaml`, runs a forward with
   `return_edges=True` on a random 2×3×512×512 tensor, asserts shapes:
   `out=[2,2,512,512]`, `edges` is a list of 3 tensors each `[2,1,512,512]`, and that
   `loss.backward()` populates grads on `edge_heads[0].conv.weight`.

---

## 4. Test & Training Protocol (run in this order)

| # | What | Command / check | Pass criterion |
|---|---|---|---|
| T1 | Unit/debug script | `python tools_edge_debug.py` | all asserts pass; edge-GT PNGs look like thin bands on tamper boundaries |
| T2 | Regression: baseline unaffected | `python test.py -g 0 -exp trufor_ph3 -in <5 sample imgs> -out /tmp/regress` on the **unmodified config** with existing weights, diff the `.npz` maps against pre-change outputs | bit-identical (code paths untouched when `edge_head` absent) |
| T3 | Overfit smoke | `mydata_ph2_edge` with `TRAIN.NUM_SAMPLES 16 TRAIN.END_EPOCH 3` via `opts` overrides | total loss and (log it once per epoch) EDG component both decrease |
| E1 | Record baseline | existing `weights/mydata_ph2/best.pth.tar` validation metrics | note `avg_p-F1_smooth`, `IoU_1_smooth` |
| E2 | Full edge run | `python train.py -exp mydata_ph2_edge -g 0` | training completes; `best.pth.tar` written |
| E3 | Compare E2 vs E1 | same validation split | **gate: `IoU_1_smooth` ≥ baseline + 0.01 and `avg_p-F1_smooth` not worse than baseline − 0.005** |
| E4 | Phase 3 on top | `python train.py -exp trufor_ph3_edge -g 0` | `avg_det_bacc` ≥ the existing Phase-3 run's value |
| A1 | (optional ablation) | E2 with `LOSS.LOSSES` EDG weight 0.4; and with `EDGE_SCALE_WEIGHTS [1.0,1.0,1.0]` | keep whichever wins E3's metric |

If E3's gate fails after trying A1: report the numbers honestly and stop — do not keep
tuning past the ablation row; the negative result is itself the answer.

---

## 5. Known Pitfalls (read before coding)

1. **`LOSS.SMOOTH: 0` in the YAMLs must not leak into `EdgeDiceLoss`.** The edge GT is
   all-zero for every pristine image; with smooth=0 dice loss saturates at ~1.0 and the
   branch would push all edge predictions to zero everywhere. `EdgeDiceLoss` hardcodes
   `smooth=1.0` — keep it that way.
2. **Do not compute edge GT in the dataset.** Albumentations transforms + random crop +
   pad(-1) happen after `__getitem__` composition
   ([data_core.py](TruFor_train_test/dataset/data_core.py)); a precomputed edge map would
   desync from the mask. On-the-fly from `labels` is always consistent.
3. **DataParallel gather:** returning `edges` as a Python list of 3 tensors is fine
   (gather recurses into lists), but never return `None` *inside* the list.
4. **Frozen loc_head in Phase 3:** `_c*` features are produced inside `torch.no_grad()`
   ([builder_np_conf.py:225-228](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L225-L228)),
   so edge-head grads can't flow into the decoder there. That's why Phase 3 does not use
   `'EDG'` — don't "fix" this by unfreezing.
5. **Resume assertion:** `train.py` asserts `checkpoint['best_key'] == best_key` on
   RESUME ([train.py:144](TruFor_train_test/train.py#L144)). New experiments get fresh
   `weights/<exp>/` dirs via the new YAML names, so no conflict — do not reuse the old
   experiment names.
6. **Optimizer param groups:** `group_weight` in `cmx/init_func.py` walks all modules;
   verify after Step 3 that the count of parameters across param groups equals
   `len(list(model.parameters()))` (the debug script's grad check catches this too).
7. **Validation images have arbitrary sizes** (batch 1, no crop). The edge logits are
   upsampled to the *input rgb* size in `encode_decode`, while labels can differ —
   `EdgeDiceLoss` handles the mismatch with its own interpolate; don't remove that.
8. **ONNX export** (`export_trufor_to_onnx.py` at repo root) calls the default forward →
   4-tuple → unaffected. Do not add `return_edges` there.
9. Masks in `MyData` are binarized `{0,1}` with `-1` padding only — if a future dataset
   uses 255-valued masks, `(target == 1)` in `_mask_to_edge_band` is the place that
   assumes binarization.

---

## 6. Deliverables Checklist

- [ ] `lib/models/cmx/ecem.py` (new)
- [ ] `lib/models/cmx/decoders/MLPDecoder.py` (modified, backward-compatible)
- [ ] `lib/models/cmx/builder_np_conf.py` (modified: modules_list, edge_heads, init/freeze, encode_decode, forward)
- [ ] `lib/core/criterion.py` (+ `EdgeDiceLoss`)
- [ ] `lib/utils.py` (get_criterion 4-tuple, FullModel EDG branch)
- [ ] `lib/config/default.py` (+ `EDGE_RADIUS`, `EDGE_SCALE_WEIGHTS`)
- [ ] `lib/config/mydata_ph2_edge.yaml` (new)
- [ ] `lib/config/trufor_ph3_edge.yaml` (new)
- [ ] `tools_edge_debug.py` (new) + `debug_edge_gt/` sample renders
- [ ] T1–T3 pass; E1–E4 numbers reported in a short results note
      (`docs/EDGE_BRANCH_RESULTS.md`), including the failure case if the E3 gate is not met

---

## 7. Exploration Roadmap — Everything Else Worth Taking From DiffForensics

The user has time and wants to try as much of DiffForensics as is sensibly portable.
These tracks are **ordered by (payoff ÷ cost)** — run them in this order, one at a time,
and record every run in the log (§7.7). Each track has a **go/no-go gate**; when a gate
fails, log it and move to the next track instead of tuning indefinitely (max 2 retune
attempts per track).

```
Track A  ECEM edge branch          (Sections 1–6)      — prerequisite for B, D
Track B  Robustness benchmark      (no training)       — measures A; reusable for all tracks
Track C  Loss & optimizer recipe   (config + 1 class)  — independent ablations
Track D  Edge-aware detection head (TruFor-only idea)  — needs A's ph2-edge checkpoint
Track E  Simplex P_sq modality     (research spike)    — the diffusion idea, scaled down
Track F  Full DDPM pre-training    (parked)            — only if E succeeds AND big GPU appears
```

### 7.1 Track B — Robustness Benchmark Harness (paper Fig. 5, adapted)

**Why:** DiffForensics' strongest empirical claim is robustness under social-media
laundering (JPEG recompression, Gaussian noise). Our Aadhaar images arrive via
WhatsApp/scan pipelines, so this is *the* deployment-relevant metric — and we currently
have no way to measure whether any change helps or hurts under laundering. Build it once,
use it to judge every track.

**What to build:** `TruFor_train_test/robustness_eval.py`:
1. Input: a checkpoint path (`TEST.MODEL_FILE` override), the MyData **validation** list,
   a perturbation spec.
2. Perturbations (applied to the decoded RGB before `/256.0`, one at a time):
   - JPEG re-encode at quality `[100, 90, 80, 70, 60, 50]` (PIL `save(..., quality=q)`);
   - additive Gaussian noise, σ ∈ `[5, 10, 15, 20, 25, 30]` on the 0–255 scale, clipped.
3. For each (perturbation, level): run inference (reuse the `test.py` forward logic —
   import, don't copy), compute per-image **pixel-F1@0.5** and **pixel-AUC** against the
   GT mask, plus image-level detection score if the checkpoint has a det head.
4. Output: one CSV per run (`robustness_<exp>.csv`: columns
   `perturb,level,avg_f1,avg_auc,avg_det_bacc`) and a matplotlib PNG with the two decay
   curves (F1 vs level), overlaying every checkpoint passed on the CLI.

**Gate (informational, never blocking):** report the curves for E1 (baseline) vs E2
(edge). Expectation from the paper: the edge model should degrade *no faster* than
baseline; if it degrades slower, that's a headline result for the writeup.

**Cost:** ~half a day of coding, inference-only compute.

### 7.2 Track C — Training-Recipe Transplants (two independent ablations)

#### C1. DiffForensics' segmentation loss (`L_seg`) as an alternative LOC criterion

Paper recipe (Eq. 4–6): `L_seg = 0.1 · weighted-BCE + 0.9 · dice`, with per-pixel weights
λ1 = 2 (tampered) and λ2 = 0.5 (pristine). TruFor's `dice_entropy` is
`0.3·CE + 0.7·dice` — close, but the CE mix is 3× heavier and the class weighting comes
from `CLASS_WEIGHTS` instead. Worth one head-to-head run.

Implementation — new class in `lib/core/criterion.py`, registered in
`get_criterion` under the name `'wbce_dice'` for `LOC`:

```python
class WBCEDiceLoss(nn.Module):
    """DiffForensics L_seg (Eq. 4-6): lam0 * weighted-BCE + (1-lam0) * binary dice.
    Operates on the 2-class softmax forged-probability channel."""
    def __init__(self, ignore_label=-1, lam0=0.1, lam1=2.0, lam2=0.5,
                 smooth=1.0, exponent=2):
        super(WBCEDiceLoss, self).__init__()
        self.ignore_label = ignore_label
        self.lam0, self.lam1, self.lam2 = lam0, lam1, lam2
        self.smooth, self.exponent = smooth, exponent

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear')
        prob = F.softmax(score, dim=1)[:, 1]                     # forged prob, B x H x W
        valid = (target != self.ignore_label).float()
        y = (target == 1).float()
        eps = 1e-6
        wbce = -(self.lam1 * y * torch.log(prob + eps)
                 + self.lam2 * (1 - y) * torch.log(1 - prob + eps))
        wbce = (wbce * valid).sum() / valid.sum().clamp(min=1)
        # reuse the same masked binary dice as EdgeDiceLoss
        p = prob.reshape(prob.shape[0], -1); t = y.reshape(y.shape[0], -1)
        v = valid.reshape(valid.shape[0], -1)
        num = (p * t * v).sum(dim=1) * 2 + self.smooth
        den = (p.pow(self.exponent) * v + t.pow(self.exponent) * v).sum(dim=1) \
              + max(self.smooth, 1e-5)
        dice = 1 - torch.mean(num / den)
        return self.lam0 * wbce + (1 - self.lam0) * dice
```

Run: clone `mydata_ph2_edge.yaml` → `mydata_ph2_edge_wbce.yaml` with
`['LOC', 1.0, 'wbce_dice']`. **Gate:** beats E2's `IoU_1_smooth`. Note: `CLASS_WEIGHTS`
is unused by this criterion (λ1/λ2 replace it) — expected, not a bug.

#### C2. AdamW optimizer (paper's choice) vs TruFor's SGD

Add to `get_optimizer` in [lib/utils.py:139-151](TruFor_train_test/lib/utils.py#L139-L151):

```python
    elif config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(params_list,
                                      lr = config.TRAIN.LR,
                                      betas = (0.9, 0.999),
                                      weight_decay = config.TRAIN.WD)
```

Run: best-so-far ph2-edge config with `TRAIN.OPTIMIZER adamw TRAIN.LR 0.0001
TRAIN.WD 0.01` (paper: AdamW, 1e-4). The poly-decay `adjust_learning_rate` still applies
on top — that's fine, it just scales the base LR. **Gate:** same as C1. Keep whichever
optimizer wins for all later tracks.

**Cost:** each ablation = one Phase-2 training run; code is <1 hour.

### 7.3 Track D — Edge-Aware Detection Head (our own extension, beyond the paper)

**Why:** TruFor's detector eats 8 statistics pooled from (conf, loc) maps
([builder_np_conf.py:249-254](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L249-L254)).
After Track A we have a third map — the finest edge map — which for card forgeries is
plausibly the most discriminative signal an image-level detector could want (real cards:
no tamper edges anywhere). DiffForensics never tries this because it has no
statistics-pooling detector; it's a TruFor-native way to exploit their module.

**Implementation:**
1. New detection mode `'confpool_edge'` in `builder_np_conf.py`:
   - constructor: when `self.conf_detection == 'confpool_edge'`, require
     `'edge_head' in MODULES` and build the detector MLP with `in_features=12`;
   - `encode_decode`: compute edges whenever
     `self.conf_detection == 'confpool_edge'` (even if `return_edges` is False —
     refactor `need_edges` to `need_edges = (return_edges or self.conf_detection ==
     'confpool_edge') and (self.edge_heads is not None)`), then:

```python
                f3 = weighted_statistics_pooling(edges[0],
                                                 F.logsigmoid(conf)).view(out.shape[0],-1)
                det = self.detection(torch.cat((f1,f2,f3),-1))
```

2. New config `trufor_ph3_edgedet.yaml` = `trufor_ph3_edge.yaml` but
   `DETECTION: 'confpool_edge'` and `PRETRAINING: 'weights/mydata_ph2_edge/best.pth.tar'`
   (the ph2 checkpoint has no `detection.*` keys, so the 12-input MLP initializes fresh —
   the existing strict=False load handles it).
3. Caution: `edges[0]` is produced from frozen-loc-head features under `no_grad` in
   Phase 3 — the edge map is a constant input to the detector there, which is exactly
   what we want (only `detection.*` trains).

**Gate:** `avg_det_bacc` > E4's value on the same validation split. Also check
`avg_det_tnr` specifically — the hoped-for effect is fewer false alarms on pristine cards.

**Cost:** ~1 day code+debug, one Phase-3 run (Phase 3 is cheap — only the MLP trains).

### 7.4 Track E — Simplex Reconstruction-Error Map (`P_sq`) as a Modality (research spike)

**Why:** this is the *actual diffusion idea* of the paper, scaled to our compute. Their
Fig. 3 shows that after Simplex-noise denoising training, the per-pixel reconstruction
error `P_sq = (x0 − x̂0)²` lights up tampered regions by itself. If that reproduces on
Aadhaar data, `P_sq` is a NP++-like evidence map we can feed TruFor — attacking exactly
the case where NP++ weakens (heavily laundered/recompressed uploads).

**Phase E-1 — the spike (go/no-go before any TruFor integration):**
1. Implement Simplex octave noise (`simplex_noise.py`): package `opensimplex` or a
   vectorized numpy port; paper params: starting frequency ν = 2⁻⁶, octaves N = 6,
   persistence γ = 0.8; normalize to zero mean / unit variance per channel so it can
   substitute ε in the standard DDPM update `x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε`.
2. Small denoiser: frozen SegFormer mit_b2 encoder (reuse
   `pretrained_models/segformers/mit_b2.pth`, single stream) + a light UNet-style conv
   decoder (4 upsampling blocks, 512→256→128→64→3, skip connections from the 4 encoder
   stages, sinusoidal time embedding added at each decoder block). Predict ε; MSE loss
   (paper Alg. 1). New standalone folder `diffusion_spike/` — do **not** touch TruFor
   model code for this phase.
3. Train on MyData images only (both pristine and tampered, **no labels**), 256×256
   random crops, batch 8, AdamW 1e-4, T=1000 cosine/linear schedule, 30–50 epochs.
   This fits a single consumer GPU (that's the point of the small decoder + 256² crops).
4. Evaluate like paper Fig. 3: for ~10 val images (mix tampered/pristine), fix t = 5
   and t = 50, compute `x̂0` from the model's ε-prediction in one step, render the grid
   `x0 | x_t | x̂0 | P_sq | GT mask` to `diffusion_spike/vis/`.

**Go/no-go gate:** on visual inspection, `P_sq` is clearly brighter inside tampered
regions than pristine background for **≥ 5 of 10** tampered samples, and does *not* light
up wholesale on pristine cards. If no-go: write up the negatives, stop Track E, skip F.

**Phase E-2 — integration (only on go):**
1. Precompute `P_sq` offline for every MyData image at fixed t (whichever of 5/50 looked
   better), save as single-channel `.npy` next to the masks; normalize per-image to
   [0,1] (log-scale first: `log1p(P_sq)` then min-max — raw squared errors are heavy-tailed).
2. Minimal-surgery integration (mirrors how NP++ enters the network):
   - dataset (`dataset_MyData.py` + `AbstractDataset._create_tensor`): optionally load
     the `.npy`, apply the *same* albumentations spatial transform + crop as the mask,
     return it stacked as extra channels of the rgb tensor's companion — concretely,
     return a 4-channel tensor and split in the model; **or** simpler and preferred:
     add a parallel `t_psq` tensor to the dataset return and thread it through
     `FullModel.forward(labels, rgbs, psqs)`. Choose the simpler path and keep it behind
     `DATASET.WITH_PSQ: false` default.
   - model: new mode `MODS: ('RGB','PSQ')` — in `EncoderDecoder.forward`, when `'PSQ'`
     is in mods, skip DnCNN and set `modal_x = torch.tile(psq, (3,1,1))` (same tiling
     convention as NP++'s 1-channel output,
     [builder_np_conf.py:276-277](TruFor_train_test/lib/models/cmx/builder_np_conf.py#L276-L277)).
3. Two ablation runs (Phase 2, edge branch kept on):
   - **E-2a `('RGB','PSQ')`** — P_sq *replaces* NP++ (tests: is P_sq at least as good?)
   - **E-2b `('RGB','NPPSQ')`** — P_sq *combined* with NP++: `modal_x =
     cat([np++ (1ch), psq (1ch), np++ (1ch)])` as a cheap 3-channel hybrid (tests: additive?)
4. Judge on standard metrics **plus Track B's laundering curves** — the hypothesis is
   specifically that P_sq degrades slower than NP++ under recompression.

**Gate:** E-2b beats the best NP++-only model on the JPEG-QF≤70 portion of the
robustness curve. **Cost:** spike ≈ 2–4 GPU-days on one consumer GPU + ~2 days coding;
integration ≈ 2 days + two Phase-2 runs.

### 7.5 Track F — Full DiffForensics Stage-1 Pre-training (PARKED)

Recorded for completeness; **do not start** unless Track E's spike is a clear "go" AND
A100-class compute becomes available. It would require: a real UNet decoder sized for
512², Simplex-DDPM pre-training ≈100 epochs (paper: 4× A100 80 GB), then rebuilding
TruFor's decoder around it and re-running Phases 2–3 from scratch. The expected marginal
gain over Track E-2b does not justify this on current hardware.

### 7.6 Order of Execution & Dependency Summary

```
A (core, §1–6) ──► B (benchmark) ──► C1 ──► C2 ──► D ──► E-1 ──go──► E-2 ──► (F parked)
                       │                                   │
                       └── rerun B's curves after every winning track
                                                           no-go ──► stop, write up
```

Rules of engagement for the implementing agent:
- One track at a time; a track's changes land as separate commits with the track letter
  in the message.
- Every track that changes model code must re-pass T1 + T2 from §4 before its training
  run starts.
- Carry winners forward: each track trains on top of the best configuration known so far
  (e.g., if C1 wins, Track D's Phase-3 sits on the wbce_dice Phase-2 checkpoint).
- Budget: max 2 retune attempts per failing gate, then log and move on.

### 7.7 Results Log (mandatory)

Maintain `docs/EXPLORATION_LOG.md` — one row per training/eval run, appended immediately
after the run finishes, including failures:

| Run ID | Track | Config / diff vs parent | Checkpoint | IoU_1_smooth | avg_p-F1_smooth | det_bacc | Robustness note | Verdict |
|---|---|---|---|---|---|---|---|---|
| E1 | – | baseline mydata_ph2 | weights/mydata_ph2/best | … | … | – | … | baseline |

Every "Verdict" is one of: `winner (carried forward)`, `no gain (dropped)`,
`failed gate (dropped)`, `blocked (reason)`. The log plus
`docs/EDGE_BRANCH_RESULTS.md` are the final deliverables of the whole exploration.
