import onnx
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

def mean_pooling(preds: np.ndarray, attention_mask: np.ndarray):
	mask_expanded = attention_mask[:, :, np.newaxis]
	return np.sum(preds * mask_expanded, 1) / mask_expanded.sum(1).clip(min=1e-9)

def normalize(x: np.ndarray, axis: int = 0, p: float = 2.):
	return x / np.linalg.norm(x, ord=p, axis=axis, keepdims=True).clip(min=1e-9)

class JinaModel:
	def __init__(self, model_path):
		self.model = onnx.load(model_path)
		self.ort_session = ort.InferenceSession(model_path)

	def predict(self, x):
		return self.ort_session.run(None, x)[0]

class JinaPipeline:
	def __init__(self, model_path: str, tokenizer_path: str):
		self.model = JinaModel(model_path)
		self.tokenizer = Tokenizer.from_file(tokenizer_path)
		self.tokenizer.enable_padding()

	def predict(self, input_strings: list[str]):
		inputs = self.tokenizer.encode_batch(input_strings)
		input_ids = np.stack([y.ids for y in inputs])
		attention_mask = np.stack([y.attention_mask for y in inputs])

		preds = self.model.predict({
			'input_ids': input_ids,
			'attention_mask': attention_mask,
		})

		output = mean_pooling(preds, attention_mask)
		output = normalize(output, axis=1)
		return output