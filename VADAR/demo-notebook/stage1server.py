# main_server.py
from time import time
from fastapi import FastAPI, UploadFile, File
import os, json, tempfile, asyncio
from full_pipeline_refactored import VADARContext  # Your class that provides qwen_generator, unik3d_model
from full_pipeline_refactored import initialize_and_get_generator, initialize_modules  # This uses the function you posted
app = FastAPI()

# Global model instance (thread-safe since no model weights are mutated)
scene_generator = None

@app.on_event("startup")
def load_all_models_once():
    global scene_generator
    context = initialize_modules(qwen_api_base_url='http://0.0.0.0:8001')
    scene_generator = initialize_and_get_generator(context)
    print("✅ Scene graph + UniK3D + LLM initialized.")

from fastapi import UploadFile, File
import json, tempfile, asyncio

@app.post("/infer/")
async def infer(file: UploadFile = File(...)):
    global scene_generator
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    with open(tmp_path, "r") as f:
        data_dict = json.load(f)

    # ✅ Run heavy inference in a background thread
    def _run():
        return scene_generator.process_and_refine_query(data_dict)
    # print(f"[{time.time()}] Starting inference...")
    detected_objects, refined_query = await asyncio.to_thread(_run)

    # results = [safe_serialize(obj) for obj in detected_objects]

    return {
        "refined_query": refined_query,
        "num_objects": len(detected_objects),
        # "objects": results
    }
