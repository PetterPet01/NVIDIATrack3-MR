MODULES_SIGNATURES = """
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
    object_prompt (str): Description of the object(s) to locate. 
                            Use "objects" to retrieve all detected objects.

Returns:
    List[DetectedObject]: Matching detected objects.
\"\"\"
def retrieve_objects(detected_objects, object_prompt):

\"\"\"
Answers a question about an object shown in a bounding box.

Args:
    image (image): Image of the scene.
    question (string): Question about the object in the bounding box.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.
    

Returns:
    string: Answer to the question about the object in the image.
\"\"\"
def vqa(image, question, bbox):

\"\"\"
Checks if two bounding boxes correspond to the same object.

Args:
    image (image): Image of the scene.
    bbox1 (list): A bounding box [xmin, ymin, xmax, ymax] containing object1.
    bbox2 (list): A bounding box [xmin, ymin, xmax, ymax] containing object2.

Returns:
    bool: True if object 1 is the same object as object 2, False otherwise.
\"\"\"
def same_object(image, bbox1, bbox2):

\"\"\"
Returns the width, height and length of the object in 3D real world (meters) space.

Args:
    detected_object (DetectedObject): A given detected object.

Returns:
    tuple: (width, height, length) of the object in 3D real world (meters) space.
\"\"\"
def get_3D_object_size(detected_object):

"""

MODULES_SIGNATURES_CLEVR = """
\"\"\"
Locates objects in an image. Object prompts should be 1 WORD MAX.

Args:
    image (image): Image to search.
    object_prompt (string): Description of object to locate. Examples: "spheres", "objects".
Returns:
    list: A list of x,y coordinates for all of the objects located in pixel space.
\"\"\"
def loc(image, object_prompt):

\"\"\"
Answers a question about the attributes of an object specified by an x,y coordinate.
Should not be used for other kinds of questions.

Args:
    image (image): Image of the scene.
    question (string): Question about the objects attribute to answer. Examples: "What color is this?", "What material is this?"
    x (int): X coordinate of the object in pixel space.
    y (int): Y coordinate of the object in pixel space. 
    

Returns:
    string: Answer to the question about the object in the image.
\"\"\"
def vqa(image, question, x, y):

\"\"\"
Checks if two pairs of coordinates correspond to the same object.

Args:
    image (image): Image of the scene.
    x_1 (int): X coordinate of object 1 in pixel space.
    y_1 (int): Y coordinate of object 1 in pixel space.
    x_2 (int): X coordinate of object 2 in pixel space.
    y_2 (int): Y coordinate of object 2 in pixel space.

Returns:
    bool: True if object 1 is the same object as object 2, False otherwise.
\"\"\"
def same_object(image, x_1, y_1, x_2, y_2):
"""

