import os
import atexit
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from download import ModelManager
from ann import ANNIndex

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=9100)
parser.add_argument('--storage_dir', type=str, default=os.path.expanduser('~/.cache/code-search'))
args = parser.parse_args()

model_manager = ModelManager(os.path.join(args.storage_dir, 'models'))
index_workspaces: dict[str, ANNIndex] = {}
app = FastAPI()

def get_index(model_name: str, workspace_name: str) -> ANNIndex:
	if workspace_name not in index_workspaces:
		index_workspaces[workspace_name] = ANNIndex(os.path.join(args.storage_dir, 'index', model_name, workspace_name), 'cosine', 768, 12)
	return index_workspaces[workspace_name]

@atexit.register
def close_workspaces():
	for workspace in index_workspaces.values():
		workspace.save()

class IndexRequest(BaseModel):
	input_strings: list[str]
	input_labels: list[str]

class QueryRequest(BaseModel):
	input_strings: list[str]

@app.get("/models")
def list_models() -> dict[str, bool]:
	return model_manager.list_models()

@app.get("/{model_name}/download")
def download_model(model_name: str):
	try:
		model_manager.download(model_name)
	except ValueError:
		raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

@app.get("/{model_name}/load")
def load_model(model_name: str):
	try:
		model_manager.load_model(model_name)
	except ValueError:
		raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

@app.post("/{model_name}/{workspace_name}/index")
def run_model(model_name: str, workspace_name: str, req: IndexRequest):
	if not model_manager.istype(model_name, 'embeddings'):
		raise HTTPException(status_code=400, detail=f"Model {model_name} not an embeddings model")

	updated_strings = get_index(model_name, workspace_name).check_updated(req.input_labels, req.input_strings)
	if any(updated_strings):
		input_strings = [string for string, updated in zip(req.input_strings, updated_strings) if updated]
		embeddings = model_manager.run_model(model_name, input_strings=input_strings)
		get_index(model_name, workspace_name).add(embeddings, req.input_labels, input_strings)

@app.post("/{model_name}/{workspace_name}/query")
def find_neighbors(model_name: str, workspace_name: str, req: QueryRequest):
	if not model_manager.istype(model_name, 'embeddings'):
		raise HTTPException(status_code=400, detail=f"Model {model_name} not an embeddings model")

	embeddings = model_manager.run_model(model_name, input_strings=req.input_strings)
	response = get_index(model_name, workspace_name).query(embeddings, thresh=0.7)
	return response

@app.delete("/{model_name}/{workspace_name}")
def clear_index(model_name: str, workspace_name: str):
	get_index(model_name, workspace_name).clear()
	del index_workspaces[workspace_name]

@app.delete("/{model_name}/{workspace_name}/{file_name}")
def delete_file(model_name: str, workspace_name: str, file_name: str):
	get_index(model_name, workspace_name).delete_file(file_name)

@app.put("/{model_name}/{workspace_name}/{old_file_name}")
def rename_file(model_name: str, workspace_name: str, old_file_name: str, new_file_name: str):
	get_index(model_name, workspace_name).rename_file(old_file_name, new_file_name)

@app.delete("/{model_name}")
def unload_model(model_name: str) -> None:
	try:
		model_manager.unload_model(model_name)
	except ValueError:
		raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")

if __name__ == "__main__":
	import uvicorn
	import psutil
	import os
	p = psutil.Process(os.getpid())
	if os.name == 'nt':
		p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
	else:
		p.nice(5)
	uvicorn.run(app, host='127.0.0.1', port=args.port)
