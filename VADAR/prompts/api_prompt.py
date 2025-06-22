API_PROMPT = """
You are an expert at implementing methods accoring to a given docstring and signature.
Implement a method given a docstring and method signature, using the API as necessary.

API:
{predef_signatures}

{generated_signatures}

Here are some examples of how to implement a method given its docstring and signature:

<docstring>
\"\"\"
Gets the material of the given object.

Args:
    image (IMAGE): Image that the object is contained in.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.

Returns:
    str: Material of the object.
\"\"\"
</docstring>
<signature>def object_material(image, bbox):</signature>
<implementation>
material = vqa(image=image, question='What material is this object?', bbox=bbox)
return material
</implementation>

<docstring>
\"\"\"
Checks if an object 1 is in front of object 2.

Args:
    image (IMAGE): Image that the object is contained in.
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
    image (IMAGE): Image that the object is contained in.
    base_object (str): Description of base object.
    relation (str): Relation to evaluate (e.g. "next to", "on top of", "on")
    target_bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the target object.

Returns:
    bool: True if the target object and the base object share the relation.
\"\"\"
</docstring>
<signature>def evaluate_object_relation(image, base_object, relation, target_bbox):</signature>
<implementation>
query = 'Is this object ' + relation + 'to ' + base_object + '?'
return vqa(image=image, question=query, target_bbox)
</implementation>


Here are some helpful definitions:
1) 2D distance/size refers to distance/size in pixel space.
2) 3D distance/size refers to distance/size in the real world. 3D size is equal to 2D size times the depth of the object.
3) We define (width, height, length) as the values along the (x, y, z) axis. Width = x axis, height = y axis, length = z axis.
4) "Depth" measures distance from the camera in 3D.

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have an initialized variable named "image" - no need to initialize it yourself.
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.
4) Do not assume that the objects you see in these questions are all of the objects you will see, keep the methods general.
5) If two objects have the same 2D width, then the object with the largest depth has the largest 3D width.
6) If two objects have the same 2D height, then the object with the largest depth has the largest 3D height.
7) 2D sizes convey the height and width in IMAGE SPACE. To convert to height and width in 3D space, it needs to be multiplied by the depth!
8) Some questions may present hypothesis measurements (e.g. "X is 3.5m wide"), these are hypothesis measurements and should be used ONLY to scale your outputs accordingly.
9) Do NOT round your answers! Always leave your answers as decimals. If the question asks "how many X do you need to get to Y" you should NOT round - leave your answer as a floating point division.
10) To determine if an object is in another object - use VQA. For example to determine if a book is in the shelf vqa(image, 'Is the book in the shelf?', book_bbox)
11) You may want to extract the 2D bounding boxes from the list of DetectedObject returned by retrieve_objects() before using them in other functions.
12) When answering with the global <regionX> tag of an object as the result, use DetectedObject.description.


Do not define new methods here, simply solve the problem using the existing methods.

Now, given the following docstring and signature, implement the method, using the API specification as necessary. Output the implementation inside <implementation></implementation>.

Again, Output the implementation inside <implementation></implementation>.

<docstring>
{docstring}
</docstring>
<signature>{signature}</signature>
"""

API_PROMPT_GQA = """
You are an expert at implementing methods accoring to a given docstring and signature.
Implement a method given a docstring and method signature, using the API as necessary.

Current API:
{predef_signatures}

{generated_signatures}

Do not define new methods here, simply solve the problem using the existing methods.

Now, given the following docstring and signature, implement the method, using the API specification as necessary. Output the implementation inside <implementation></implementation>.

Again, Output the implementation inside <implementation></implementation>.

<docstring>
{docstring}
</docstring>
<signature>{signature}</signature>
"""

API_PROMPT_CLEVR = """
Implement a method given a docstring and method signature, using the API specification as necessary.

Current API:
{predef_signatures}

{generated_signatures}

Here are some examples of how to implement a method given its docstring and signature:

<docstring>
\"\"\"
Locates objects that are on the left of the reference object.

Args:
    image (IMAGE): Image to search.
    ref_x (int): X coordinate of reference object in pixel space.
    ref_y (int): Y coordinate of reference object in pixel space.

Returns:
    points (list): list of [x, y] coordinates for objects in pixel space matching description to the left.
\"\"\"
</docstring>
<signature>def objects_left(image, ref_x, ref_y):</signature>
<implementation>
objects_left = []
all_objects = retrieve_objects(detected_objects, object_prompt='objects')
for detected_object in all_objects:
    object_bbox = extract_2d_bounding_box(detected_object)
    object_point = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    x, y = object_point
    if same_object(image, ref_x, ref_y, x, y):
        continue
    if x < ref_x:
        objects_left.append(object_point)
return objects_left
</implementation>

<docstring>
\"\"\"
Gets the material of the given object.

Args:
    image (IMAGE): Image that the object is contained in.
    ref_x (int): X coordinate of reference object in pixel space.
    ref_y (int): Y coordinate of reference object in pixel space.

Returns:
    str: Material of the object.
\"\"\"
</docstring>
<signature>def object_material(image, ref_x, ref_y):</signature>
<implementation>
material = vqa(image=image, question='What material is this object?', x=ref_x, y=ref_y)
return material
</implementation>

<docstring>
\"\"\"
Checks if an object 1 is in front of object 2.

Args:
    image (IMAGE): Image that the object is contained in.
    x_1 (int): X coordinate of object 1 in pixel space.
    y_1 (int): Y coordinate of object 1 in pixel space.
    x_2 (int): X coordinate of object 2 in pixel space.
    y_2 (int): Y coordinate of object 2 in pixel space.

Returns:
    bool: True if object 1 is in front of object 2, False otherwise
\"\"\"
</docstring>
<signature>def in_front_of(image, x_1, y_1, x_2, y_2):</signature>
<implementation>
depth_1 = depth(image, x_1, y_1)
depth_2 = depth(image, x_2, y_2)
return depth_1 < depth_2
</implementation>

<docstring>
\"\"\"
Checks if object1 has the same size as object2

Args:
    image (IMAGE): Image that the object is contained in.
    x_1 (int): X coordinate of object 1 in pixel space.
    y_1 (int): Y coordinate of object 1 in pixel space.
    x_2 (int): X coordinate of object 2 in pixel space.
    y_2 (int): Y coordinate of object 2 in pixel space.

Returns:
    bool: True if object 1 has the same size as object 2, False otherwise
\"\"\"
</docstring>
<signature>def same_size(image, x_1, y_1, x_2, y_2):</signature>
<implementation>
object_1_size = vqa(image=image, question='What size is this object?', x=x_1, y=y_1)
object_2_size = vqa(image=image, question='What size is this object?', x=x_2, y=y_2)
return object_1_size == object_2_size
</implementation>

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have 2 initialized variables named "image" and "detected_objects" - no need to initialize them yourself! 
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.
4) You may want to extract the 2D bounding boxes from the list of DetectedObject returned by retrieve_objects() before using them in other functions.

Do not define new methods here, simply solve the problem using the existing methods.

Now, given the following docstring and signature, implement the method, using the API specification as necessary. Output the implementation inside <implementation></implementation>.

Again, Output the implementation inside <implementation></implementation>.

<docstring>
{docstring}
</docstring>
<signature>{signature}</signature>
"""

API_PROMPT_VSI = """
Implement a method given a docstring and method signature, using the API specification as necessary.

Current API:
{predef_signatures}

{generated_signatures}

Here are some examples of how to implement a method given its docstring and signature:

<docstring>
\"\"\"
Locates objects that are on the left of the reference object.

Args:
    image (IMAGE): Image to search.
    ref_x (int): X coordinate of reference object in pixel space.
    ref_y (int): Y coordinate of reference object in pixel space.

Returns:
    points (list): list of [x, y] coordinates for objects in pixel space matching description to the left.
\"\"\"
</docstring>
<signature>def objects_left(image, ref_x, ref_y):</signature>
<implementation>
objects_left = []
all_objects = retrieve_objects(detected_objects, object_prompt='objects')
for detected_object in all_objects:
    object_bbox = extract_2d_bounding_box(detected_object)
    object_point = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    x, y = object_point
    if same_object(image, ref_x, ref_y, x, y):
        continue
    if x < ref_x:
        objects_left.append(object_point)
return objects_left
</implementation>

<docstring>
\"\"\"
Gets the material of the given object.

Args:
    image (IMAGE): Image that the object is contained in.
    ref_x (int): X coordinate of reference object in pixel space.
    ref_y (int): Y coordinate of reference object in pixel space.

Returns:
    str: Material of the object.
\"\"\"
</docstring>
<signature>def object_material(image, ref_x, ref_y):</signature>
<implementation>
material = vqa(image=image, question='What material is this object?', x=ref_x, y=ref_y)
return material
</implementation>

<docstring>
\"\"\"
Checks if an object 1 is in front of object 2.

Args:
    image (IMAGE): Image that the object is contained in.
    x_1 (int): X coordinate of object 1 in pixel space.
    y_1 (int): Y coordinate of object 1 in pixel space.
    x_2 (int): X coordinate of object 2 in pixel space.
    y_2 (int): Y coordinate of object 2 in pixel space.

Returns:
    bool: True if object 1 is in front of object 2, False otherwise
\"\"\"
</docstring>
<signature>def in_front_of(image, x_1, y_1, x_2, y_2):</signature>
<implementation>
depth_1 = depth(image, x_1, y_1)
depth_2 = depth(image, x_2, y_2)
return depth_1 < depth_2
</implementation>

<docstring>
\"\"\"
Checks if object1 has the same size as object2

Args:
    image (IMAGE): Image that the object is contained in.
    x_1 (int): X coordinate of object 1 in pixel space.
    y_1 (int): Y coordinate of object 1 in pixel space.
    x_2 (int): X coordinate of object 2 in pixel space.
    y_2 (int): Y coordinate of object 2 in pixel space.
    epsilon (float): Acceptable margin of error in sizes.

Returns:
    bool: True if object 1 has the same size as object 2, False otherwise
\"\"\"
</docstring>
<signature>def same_size(image, x_1, y_1, x_2, y_2, epsilon):</signature>
<implementation>
object_1_height, object_1_width = get_2D_object_size(image, x_1, y_1)
object_2_height, object_2_width = get_2D_object_size(image, x_2, y_2)
return abs(object_1_height - object_2_height) < epsilon and abs(object_1_width - object_2_width) < epsilon
</implementation>

Here are some helpful definitions:
1) 2D distance/size refers to distance/size in pixel space.
2) 3D distance/size refers to distance/size in the real world. 3D size is equal to 2D size times the depth of the object.
3) "On" is defined as the closest object ABOVE another object. Only use this definition for "on".
4) "Next to" is defined as the closest object.
5) Width is the same as length.
6) "Depth" measures distance from the camera in 3D.

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have an initialized variable named "image" - no need to initialize it yourself! 
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.
4) Do not assume that the objects you see in these questions are all of the objects you will see, keep the methods general.
5) If two objects have the same 2D width, then the object with the largest depth has the largest 3D width.
6) If two objects have the same 2D height, then the object with the largest depth has the largest 3D height.
7) 2D sizes convey the height and width in IMAGE SPACE. To obtain absolute measures of height and width in 3D space, you need to multiply the 2D size by the depth and divide by the focal length!
8) If you are given a reference size, scale your output predicted size accordingly!
9) PAY ATTENTION TO UNITS. Do not assume things will be in either centimeters or meters. The question will tell you which unit to use. Scale your outputs accordingly.
10) A pixel (x,y) on the image can be converted to a 3D point (Z * (x - px)/f, Z * (y - py)/f, Z) whrere f is the focal length, Z is the depth, and (px, py) is the principal point. You can assume the principal point is at the center of the image (px = W/2, py = H/2). You can access the (width, height) of the image by doing image.size

Do not define new methods here, simply solve the problem using the existing methods.

Now, given the following docstring and signature, implement the method, using the API specification as necessary. Output the implementation inside <implementation></implementation>.

Again, Output the implementation inside <implementation></implementation>.

<docstring>
{docstring}
</docstring>
<signature>{signature}</signature>
"""
