from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# --- API Request Models ---

class ImageUrl(BaseModel):
    url: str # Expected format: "data:{mime_type};base64,{base64_string}"

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ImageOptions(BaseModel):
    # For providing pre-computed segmentation masks
    provided_seg_masks: Optional[List[str]] = Field(None, description="List of base64 encoded PNG strings for segmentation masks.")
    
    # For generating segmentation from boxes on the server
    regions_boxes: Optional[List[List[int]]] = Field(None, description="List of bounding boxes [[x1,y1,x2,y2], ...].")
    use_sam_segmentation: bool = Field(True, description="If true and boxes are provided, run SAM to get masks.")
    
    # For providing a pre-computed depth map
    depth_image_url: Optional[ImageUrl] = None
    process_provided_depth: bool = Field(False, description="If true, use the provided depth_image_url.")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    conv_mode: str = "llama_3"
    use_bfloat16: bool = True
    image_options: Optional[ImageOptions] = None


# --- API Response Models (OpenAI-compatible) ---

class ResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ResponseChoice]
    usage: UsageInfo