import os

import torch

from lib.config import config
from lib.utils import get_model


class TruForOnnxWrapper(torch.nn.Module):
	"""Wrapper to freeze save_np=False and expose stable ONNX outputs."""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, rgb):
		pred, conf, det, _ = self.model(rgb, save_np=False)
		if conf is None:
			conf = torch.zeros(
				(pred.shape[0], 1, pred.shape[2], pred.shape[3]),
				dtype=pred.dtype,
				device=pred.device,
			)
		if det is None:
			det = torch.zeros((pred.shape[0], 1), dtype=pred.dtype, device=pred.device)
		return pred, conf, det


def save_state_dict(checkpoint_path, state_dict_out):
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	if "state_dict" not in checkpoint:
		raise KeyError(
			"Checkpoint does not contain 'state_dict'. "
			f"Found keys: {list(checkpoint.keys())}"
		)

	os.makedirs(os.path.dirname(state_dict_out) or ".", exist_ok=True)
	torch.save(checkpoint["state_dict"], state_dict_out)
	print(f"Saved state_dict to: {state_dict_out}")
	return checkpoint["state_dict"]


def build_model_from_config(trufor_config, state_dict, device):
	config.defrost()
	config.merge_from_file(trufor_config)
	config.freeze()

	model = get_model(config)
	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"[WARN] Missing keys: {len(missing)}")
	if unexpected:
		print(f"[WARN] Unexpected keys: {len(unexpected)}")

	model = model.to(device)
	model.eval()
	return model


def export_to_onnx(model, onnx_out, opset, height, width, dynamic, device):
	wrapper = TruForOnnxWrapper(model).to(device).eval()
	dummy = torch.randn(1, 3, height, width, dtype=torch.float32, device=device)

	dynamic_axes = None
	if dynamic:
		dynamic_axes = {
			"rgb": {0: "batch", 2: "height", 3: "width"},
			"pred": {0: "batch", 2: "height", 3: "width"},
			"conf": {0: "batch", 2: "height", 3: "width"},
			"det": {0: "batch"},
		}

	os.makedirs(os.path.dirname(onnx_out) or ".", exist_ok=True)
	with torch.no_grad():
		torch.onnx.export(
			wrapper,
			dummy,
			onnx_out,
			export_params=True,
			opset_version=opset,
			do_constant_folding=True,
			input_names=["rgb"],
			output_names=["pred", "conf", "det"],
			dynamic_axes=dynamic_axes,
			dynamo=False,
		)

	print(f"Saved ONNX model to: {onnx_out}")


# Hardcoded settings for direct run (no argparse)
CHECKPOINT_PATH = "/sharedrive/Tushar Thoriya/GitHub techniques/TruFor/API_setup/weights/custom_data_train_eph_100.pth.tar"
TRUFOR_CONFIG = "lib/config/trufor_ph3.yaml"
STATE_DICT_OUT = "weights/custom_data_train_eph_100_state_dict.pth"
ONNX_OUT = "weights/tamper_detection_trufor_20260312.onnx"
OPSET = 17
INPUT_HEIGHT = 768
INPUT_WIDTH = 768
DYNAMIC_AXES = True
EXPORT_DEVICE = "cuda"  # "cpu", "cuda", or "auto"


def main():
	device_mode = EXPORT_DEVICE.strip().lower()
	if device_mode == "cpu":
		device = torch.device("cpu")
	elif device_mode == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("EXPORT_DEVICE is 'cuda' but CUDA is not available.")
		device = torch.device("cuda")
	elif device_mode == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		raise ValueError("EXPORT_DEVICE must be one of: 'cpu', 'cuda', 'auto'.")

	print(f"Using device: {device}")

	state_dict = save_state_dict(CHECKPOINT_PATH, STATE_DICT_OUT)
	model = build_model_from_config(TRUFOR_CONFIG, state_dict, device)
	export_to_onnx(
		model=model,
		onnx_out=ONNX_OUT,
		opset=OPSET,
		height=INPUT_HEIGHT,
		width=INPUT_WIDTH,
		dynamic=DYNAMIC_AXES,
		device=device,
	)


if __name__ == "__main__":
	main()
