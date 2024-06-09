from dataclasses import dataclass
import os
from typing import Literal
import requests
from tqdm import tqdm

def download_file(url: str, filename: str, chunk_size: int = 1024):
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		total_size = int(r.headers.get('content-length', 0))

		with open(filename, 'wb') as f:
			with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filename)) as pbar:
				for chunk in r.iter_content(chunk_size=chunk_size):
					if chunk:
						f.write(chunk)
						pbar.update(len(chunk))

@dataclass
class Model:
	name: str
	type: Literal['embeddings']
	files: list[str]

class ModelManager:
	models: dict[str, Model] = {
		'jina': Model(
			name='jina',
			type='embeddings',
			files=[
			'https://huggingface.co/jinaai/jina-embeddings-v2-base-code/resolve/main/onnx/model_quantized.onnx?download=true',
			'https://huggingface.co/jinaai/jina-embeddings-v2-base-code/resolve/main/tokenizer.json?download=true'
		])
	}

	def __init__(self, storage_dir: str):
		self.storage_dir = storage_dir
		self.loaded_models = {}
		os.makedirs(storage_dir, exist_ok=True)

	def istype(self, model_name: str, model_type: Literal['embeddings']) -> bool:
		return self.models[model_name].type == model_type

	def relpath(self, *args) -> str:
		return os.path.join(self.storage_dir, *args)

	def list_models(self) -> dict[str, bool]:
		return {model: os.path.exists(self.relpath(model)) for model in self.models}

	def download(self, model_name: str):
		model = self.models[model_name]

		if model.name not in self.models:
			raise ValueError(f'Model {model.name} not found')
		if os.path.exists(self.relpath(model.name)):
			return

		os.mkdir(self.relpath(model.name))
		for url in model.files:
			filename = os.path.basename(url.split('?')[0])
			download_file(url, self.relpath(model.name, filename))

	def load_model(self, model_name: str):
		model = self.models[model_name]

		if model.name in self.loaded_models:
			return
		self.download(model.name)

		if model.type == 'embeddings':
			from models import JinaPipeline
			self.loaded_models[model.name] = JinaPipeline(
				model_path=self.relpath(model.name, 'model_quantized.onnx'),
				tokenizer_path=self.relpath(model.name, 'tokenizer.json')
			)
		else:
			raise ValueError(f'Model type {model.type} not found')

	def unload_model(self, model_name: str):
		if model_name not in self.loaded_models:
			raise ValueError(f'Model {model_name} not loaded')
		del self.loaded_models[model_name]

	def run_model(self, model_name: str, **kwargs):
		if model_name not in self.loaded_models:
			self.load_model(model_name)
		return self.loaded_models[model_name].predict(**kwargs)