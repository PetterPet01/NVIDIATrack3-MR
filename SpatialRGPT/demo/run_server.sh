#!/bin/bash

# --- IMPORTANT ---
# Set these environment variables to the correct paths on your system.
# The server will fail to start if these are not set correctly.
export SPATIALRGPT_MODEL_PATH="a8cheng/SpatialRGPT-VILA1.5-8B"
export SPATIALRGPT_MODEL_NAME="SpatialRGPT-VILA1.5-8B"
export DEPTH_ANYTHING_PATH="/root/Dev/SpatialRGPT/Depth-Anything-V2"
export SAM_CKPT_PATH="/root/Dev/SpatialRGPT/sam_hq_vit_h.pth"

# --- Add Project Root to PYTHONPATH ---
# This ensures that modules like 'llava' can be found.
# Adjust the path if your project structure is different.
export PYTHONPATH=${PYTHONPATH}:/root/Dev/SpatialRGPT

echo "--- Configuration ---"
echo "Model Path: $SPATIALRGPT_MODEL_PATH"
echo "Model Name: $SPATIALRGPT_MODEL_NAME"
echo "SAM Checkpoint: $SAM_CKPT_PATH"
echo "Depth Anything Path: $DEPTH_ANYTHING_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "---------------------"

# Run the FastAPI server
# It will be accessible at http://localhost:8001 to match the desired function's default.
uvicorn spatialrgpt_server.main:app --host 0.0.0.0 --port 8001