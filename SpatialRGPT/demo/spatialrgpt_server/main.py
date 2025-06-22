import time
import uuid
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .api_models import ChatCompletionRequest, ChatCompletionResponse, ResponseChoice, UsageInfo, ChatMessage
from .utils.model_loader import load_all_models
from .utils.image_processing import decode_base64_image, get_depth_map, segment_using_boxes
from .utils.inference import run_vlm_inference

# --- Globals for Models ---
# This dictionary will be populated at startup
MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load all models ---
    logging.info("Server starting up... Loading models.")
    MODELS.update(load_all_models())
    logging.info("Model loading complete. Server is ready.")
    yield
    # --- Shutdown: Clean up resources if needed ---
    logging.info("Server shutting down.")
    MODELS.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        # --- 1. Extract Main Image and Prompt ---
        last_message = request.messages[-1]
        image_content = None
        if isinstance(last_message.content, list):
            for item in last_message.content:
                if item.type == "image_url":
                    image_content = item
                    break
        
        if not image_content:
            raise HTTPException(status_code=400, detail="No image provided in the last message.")

        pil_image = decode_base64_image(image_content.image_url.url)
        raw_image_np = np.array(pil_image)

        # --- 2. Prepare Segmentation Masks ---
        seg_masks = []
        opts = request.image_options
        if opts:
            # Priority 1: Use provided segmentation masks
            if opts.provided_seg_masks:
                logging.info(f"Using {len(opts.provided_seg_masks)} provided segmentation masks.")
                for mask_b64 in opts.provided_seg_masks:
                    mask_pil = decode_base64_image(mask_b64).convert('L')
                    # VLM expects a binary mask of 0s and 1s
                    mask_np = (np.array(mask_pil) > 128).astype(np.uint8)
                    seg_masks.append(mask_np)
            
            # # Priority 2: Generate masks from boxes using SAM
            # elif opts.regions_boxes and opts.use_sam_segmentation:
            #     if MODELS.get("sam_predictor"):
            #         logging.info(f"Generating segmentation for {len(opts.regions_boxes)} boxes using SAM.")
            #         seg_masks = segment_using_boxes(raw_image_np, opts.regions_boxes, MODELS["sam_predictor"])
            #     else:
            #         logging.warning("SAM requested but not loaded. Skipping segmentation.")
        
        # --- 3. Prepare Depth Map ---
        # The VLM expects a 3-channel image for depth, even if it's grayscale.
        colorized_depth = np.zeros_like(raw_image_np, dtype=np.uint8)
        prompt_str = last_message.content if isinstance(last_message.content, str) else "".join([c.text for c in last_message.content if c.type == 'text'])

        # Case A: User provides a depth map
        if opts and opts.depth_image_url and opts.process_provided_depth:
            logging.info("Using provided depth map.")
            depth_pil = decode_base64_image(opts.depth_image_url.url)
            colorized_depth = np.array(depth_pil)
        # Case B: Prompt needs depth, so we generate it
        # elif "<depth>" in prompt_str:
        #     if MODELS.get("depth_model") and MODELS.get("depth_transform"):
        #         logging.info("Generating depth map on-the-fly.")
        #         colorized_depth = get_depth_map(raw_image_np, MODELS["depth_model"], MODELS["depth_transform"])
        #     else:
        #         logging.warning("Depth requested in prompt but model not loaded. Using blank depth map.")

        # --- 4. Run Inference ---
        logging.info("Running VLM inference...")
        assistant_response = run_vlm_inference(
            request=request,
            all_models=MODELS,
            raw_image=raw_image_np,
            seg_masks=seg_masks,
            colorized_depth=colorized_depth
        )

        # --- 5. Format Response ---
        response_message = ChatMessage(role="assistant", content=assistant_response)
        choice = ResponseChoice(index=0, message=response_message, finish_reason="stop")
        usage = UsageInfo() # Note: Token counting not implemented for simplicity
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=usage
        )
    
    except HTTPException as e:
        # Re-raise HTTP exceptions to let FastAPI handle them
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        # Return a generic 500 error for other exceptions
        raise HTTPException(status_code=500, detail=str(e))