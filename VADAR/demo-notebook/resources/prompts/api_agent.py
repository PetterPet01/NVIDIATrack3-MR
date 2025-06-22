API_PROMPT = """
You are an expert at implementing methods accoring to a given docstring and signature.
Implement a method given a docstring and method signature, using the API as necessary.

API:
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
    
{predef_signatures}

{generated_signatures}

Here are some examples of how to implement a method given its docstring and signature. Look at these FOR REFERENCE ONLY. The methods below are for illustrative purposes and can't be used in code:

<docstring>
\"\"\"
Gets if the given object is empty.

Args:
    image (PIL.Image.Image): Image that the object is contained in.
    depth (PIL.Image.Image): Depth Image for the image.
    object (DetectedObject): A detected object instance representing the object.

Returns:
    str: Material of the object.
\"\"\"
</docstring>
<signature>def is_object_empty(image, depth, detected_object):</signature>
<implementation>
is_empty = vqa(image=image, depth=depth, question='Is this object <mask> empty?', objects=[detected_object])
return is_empty
</implementation>

<docstring>
\"\"\"
Checks if an object 1 is in front of object 2.

Args:
    image (PIL.Image.Image): Image that the object is contained in.
    bbox1 (List[Tuple[float, float, float]]): A list of 8 (x, y, z) tuples, representing the corners of a 3D bounding box containing object1.
    bbox2 (List[Tuple[float, float, float]]): A list of 8 (x, y, z) tuples, representing the corners of a 3D bounding box containing object2.

Returns:
    bool: True if object 1 is in front of object 2, False otherwise
\"\"\"
</docstring>
<signature>def in_front_of(image, bbox1, bbox2):</signature>
<implementation>
A_min_z = A_points[:, 2].min()
B_min_z = B_points[:, 2].min()
return A_min_z < B_min_z
</implementation>

<docstring>
\"\"\"
Checks if a target object and a base object share a spatial relationship (e.g. "next to", "on top of", "on")

Args:
    image (PIL.Image.Image): Image that the object is contained in.
    depth (PIL.Image.Image): Depth Image of the Image.
    base_object (str): Description of base object.
    relation (str): Relation to evaluate (e.g. "next to", "on top of", "on")
    target_object (DetectedObject): A DetectedObject instance of the target object.

Returns:
    bool: True if the target object and the base object share the relation.
\"\"\"
</docstring>
<signature>def evaluate_object_relation(image, depth, base_object, relation, target_object):</signature>
<implementation>
query = 'Is this object <mask> ' + relation + 'to ' + base_object + '?'
return vqa(image=image, depth=depth, question=query, objects=[target_object])
</implementation>


Here are some helpful instructions: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have two initialized variable named "image" (of type "PIL.Image.Image") and "detected_objects" (of type list[DetectedObject]) - no need to initialize them yourself! 
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if specific objects are inside the region of a larger object with the find_overlapping_regions method.
4) Do not assume that the objects you see in these questions are all of the objects you will see, keep the methods general.
5) Do NOT round your answers! Always leave your answers as decimals. If the question asks "how many X do you need to get to Y" you should NOT round - leave your answer as a floating point division.
6) Whenever the query has visual cues (i.e. important to distinguish among similar named objects) - MUST use vqa().
7) You may want to extract the 2D bounding boxes from the list of DetectedObject returned by retrieve_objects() before using them in other functions.
8) When the query asks WHICH object is suitable, ANSWER with DetectedObject.description FOR THE TAG ONLY. The <regionX> tags in DetectedObject.description ALWAYS HAVE the '<' and '>' symbols. Please consider that when writing the code.
9) The vqa() function returns a STRING. If you want to parse the output as a boolean or numeric value, for example, please do so appropriately using is_similar_text()
10) DO NOT use simple naive math for moderately complex subtasks for the question. When asking questions that require visual understanding better than naive, unintuitive hardcoded method, use vqa().
11) ONLY generate executable code, not simply a function definition.
12) Be more flexiable about positional understanding. For example, if the query is asking the object at the rightmost position, please check the position with a function, and not naive index-based assumptions.
13) ESPECIALLY FOR COUNTING, when checking if some objects are inside another object, use the find_overlapping_regions() function to retrieve the region index (can be used with detected_objects[idx]).
14) When calculating distances between two objects, don't do 2D distance. USE the provided calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject) function from the API above.
15) DetectedObject.description ONLY denotes the type of object - NOT ANY additional relational info (i.e. 'pallet', 'transporter', 'buffer')

Do not define new methods here, simply solve the problem using the existing methods.

Now, given the following docstring and signature, implement the method, using the API specification as necessary. Output the implementation inside <implementation></implementation>.

Again, Output the implementation inside <implementation></implementation>.

<docstring>
{docstring}
</docstring>
<signature>{signature}</signature>
<question>{question}</question>
"""

