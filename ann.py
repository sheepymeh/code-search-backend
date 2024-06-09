from dataclasses import dataclass
import os
import json
import base64
from typing import Literal
import hnswlib
import numpy as np
from hashlib import sha256

@dataclass
class IndexedItem:
	label: str
	hash: int

# NOTE: hnswlib cosine similarity is 1 - cosine similarity
class ANNIndex:
	ids: list[str|None]
	hashes: list[bytes|None]
	label_to_id: dict[str, int]

	def __init__(
		self,
		path: str,
		space: Literal['cosine', 'l2'],
		dim: int,
		num_threads: int = -1
	):
		max_elements = 10_000

		self.path = path
		self.index = hnswlib.Index(space, dim)
		self.index.set_num_threads(num_threads)
		self.index.set_ef(10)

		if os.path.exists(path):
			with open(os.path.join(path, 'ids.json')) as f:
				data = json.load(f)
				self.ids = data['ids']
				self.hashes = [base64.b64decode(x) if x is not None else None for x in data['hashes']]
				self.label_to_id = data['label_to_id']
			self.index.load_index(
				os.path.join(path, 'index.bin'),
				max_elements=max_elements,
				allow_replace_deleted=True
			)
		else:
			self.ids = [None] * max_elements
			self.hashes = [None] * max_elements
			self.label_to_id = {}
			self.index.init_index(
				max_elements=max_elements,
				ef_construction=200,
				M=16,
				allow_replace_deleted=True
			)

	def save(self):
		os.makedirs(self.path, exist_ok=True)
		self.index.save_index(os.path.join(self.path, 'index.bin'))
		with open(os.path.join(self.path, 'ids.json'), 'w') as f:
			json.dump({
				'ids': self.ids,
				'hashes': [base64.b64encode(x).decode() if x is not None else None for x in self.hashes],
				'label_to_id': self.label_to_id
			}, f)

	def _register(self, label: str, value: str) -> int:
		if label not in self.label_to_id:
			new_id = self.ids.index(None)
			self.label_to_id[label] = new_id
			self.ids[new_id] = label
			self.hashes[new_id] = sha256(value.encode('utf-8')).digest()
			return new_id
		else:
			return self.label_to_id[label]

	def ids_in_file(self, file_name: str) -> list[int]:
		return [self.label_to_id[label] for label in self.label_to_id.keys() if label.startswith(f'{file_name}:')]

	def query(self, data: np.ndarray, k: int = 10, thresh: float = .5) -> list[dict[str, float]]:
		output = []
		if not self.index.get_current_count():
			return output

		ids, distances = self.index.knn_query(data, k=min(k, self.index.get_current_count()))
		for batch_ids, batch_distances in zip(ids, distances):
			mask = batch_distances < thresh
			batch_ids, batch_distances = batch_ids[mask], batch_distances[mask]
			output.append({
				self.ids[sample_id]: float(distance)
				for sample_id, distance in zip(batch_ids, batch_distances)
			})
		return output

	def check_updated(self, labels: list[str], values: list[str]) -> list[bool]:
		ids = [self.label_to_id.get(label, None) for label in labels]
		hashes = [sha256(value.encode('utf-8')).digest() for value in values]
		return [id is None or self.hashes[id] != hash for id, hash in zip(ids, hashes)]

	def add(self, data: np.ndarray, labels: list[str], values: list[str]):
		ids = np.array([self._register(label, value) for label, value in zip(labels, values)])
		for id in ids:
			try:
				self.index.unmark_deleted(id)
			except RuntimeError:
				pass
		self.index.add_items(data, ids)

	def delete(self, ids: np.ndarray|list[int]):
		for id in ids:
			del self.label_to_id[self.ids[id]]
			self.index.mark_deleted(id)
			self.ids[id] = None
			self.hashes[id] = None

	def delete_file(self, file_name: str):
		ids = self.ids_in_file(file_name)
		self.delete(np.array(ids))

	def rename_file(self, old_file_name: str, new_file_name: str):
		ids = self.ids_in_file(old_file_name)
		for id in ids:
			old_label: str = self.ids[id]
			new_label = f'{new_file_name}:{old_label[:len(old_file_name)+1]}'
			self.label_to_id[new_label] = id
			self.ids[id] = new_label
			del self.label_to_id[new_label]

	def clear(self):
		try:
			os.unlink(os.path.join(self.path, 'index.bin'))
			os.unlink(os.path.join(self.path, 'ids.json'))
			os.rmdir(self.path)
		except FileNotFoundError:
			pass

	def test(self):
		pass