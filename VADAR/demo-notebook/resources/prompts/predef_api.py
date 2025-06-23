MODULES_SIGNATURES = """
\"\"\"
Represents a detected object in both 2D and 3D spaces with associated metadata.

Args:
    class_name (str): The class label of the detected object (e.g., \"car\", \"person\").
    description (str): A textual description of the object.
    segmentation_mask_2d (np.ndarray): A binary mask array representing the 2D segmentation of the object.
    rle_mask_2d (str): Run-length encoded representation of the 2D mask.
    bounding_box_2d (np.ndarray | None): The 2D bounding box of the object, typically in [x1, y1, x2, y2] format. Optional.
    point_cloud_3d (o3d.geometry.PointCloud): The 3D point cloud corresponding to the object.
    bounding_box_3d_oriented (o3d.geometry.OrientedBoundingBox): Oriented 3D bounding box of the object.
    bounding_box_3d_axis_aligned (o3d.geometry.AxisAlignedBoundingBox): Axis-aligned 3D bounding box of the object.
    image_crop_pil (PILImage.Image | None): Optional cropped image of the object in PIL format.
\"\"\"
class DetectedObject:

\"\"\"
Compares if two texts are semantically similar.

Args:
    text1 (str): The first text to compare.
    text2 (str): The second text to compare.
Returns:
    bool: True if text1 is equal in meaning to text2, else False.
\"\"\"
def is_similar_text(text1: str, text2: str) -> bool:

\"\"\"
Extracts 2D bounding box as 2 points (top-left and bottom-right) in 2D pixel space.

Args:
    detected_object (DetectedObject): Detected object to retrieve 2D bounding box.
Returns:
    array: An array of int for the bounding box [xmin, ymin, xmax, ymax] for the object located in pixel space.
\"\"\"
def extract_2d_bounding_box(detected_object):

\"\"\"
Extracts oriented 3D bounding box 8 corner points in 3D space.

Args:
    detected_object (DetectedObject): Detected object to retrieve 3D bounding box.

Returns:
    List[Tuple[float, float, float]]: A list of 8 (x, y, z) tuples, representing the corners of a 3D bounding box.
\"\"\"
def extract_3d_bounding_box(detected_object):

\"\"\"
Returns a list of DetectedObject instances matching the given object_prompt.

Args:
    detected_objects (List[DetectedObject]): List of detected objects.
    object_prompt (str): Class name of the object(s) to locate (i.e. "pallet", "transporter", "buffer", etc.). Use "objects" to retrieve all detected objects.

Returns:
    List[DetectedObject]: Matching detected objects.
\"\"\"
def retrieve_objects(detected_objects, object_prompt):

\"\"\"
Answers a question about an object shown in a bounding box.

Args:
    image (PIL.Image.Image): Image of the scene.
    depth (PIL.Image.Image): Depth Image of the scene.
    question (string): Question about the object in the bounding box. For each DetectedObject in objects, there must a corresponding <mask> tag in the question.
    objects (list[DetectedObject]): A list of (DetectedObject) objects to consider for VLM.
    

Returns:
    string: Answer to the question about the object in the image.
\"\"\"
def vqa(image, depth, question, objects):

\"\"\"
Finds regions that significantly overlap with a parent region. MUST USE for counting objects inside a region / DetectedObject

Args:
    parent_region (DetectedObject): The main region of interest with a 2D segmentation mask and bounding box.
    countable_regions (List[DetectedObject]): A list of region objects to evaluate for overlap with the parent region.

Returns:
    List[int]: A list of int representing the region index of an overlapping region
    Only regions exceeding the overlap threshold are returned.
\"\"\"
def find_overlapping_regions(parent_region: DetectedObject, countable_regions: List[DetectedObject]) -> List[Tuple[int, float]]:

\"\"\"
Calculates the distance between two 3D objects.

Args:
    obj1 (DetectedObject): The first detected object, containing a 3D point cloud.
    obj2 (DetectedObject): The second detected object, containing a 3D point cloud.

Returns:
    float: (meters) distance representing the point cloud distance between the two objects.
\"\"\"
def calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject):

\"\"\"
Returns the width, height and length of the object in 3D real world (meters) space.

Args:
    detected_object (DetectedObject): A given detected object.

Returns:
    tuple: (width, height, length) of the object in 3D real world (meters) space.
\"\"\"
def get_3D_object_size(detected_object):

"""
