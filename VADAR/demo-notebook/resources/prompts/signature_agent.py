SIGNATURE_PROMPT = """
Propose only new method signatures to add to the existing API.
Also, You MUST at least build signatures for the methods below:
{vqa_functions}

Available Primitives: image, float, string, list, int

Current API:
{signatures}

Next, I will ask you a series of questions that reference an image and are solvable with a python program that uses the API I have provided so far. Please propose new method signatures with associated docstrings to add to the API that would help modularize the programs that answer the questions. 

For each proposed method, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>. Do not propose methods that are already in the API.

Please ensure that you ONLY add new methods when necessary. Do not add new methods if you can solve the problem with combinations of the previous methods!

Added methods should be simple, building minorly on the methods that already exist. Do NOT assume that the objects you see in these questions are all of the objects you will see, keep the methods general.

Here are some helpful instructions and definitions:
1) Do NOT round your answers! Always leave your answers as decimals even when it feels intuitive to round or ceiling your answer - do not do it!
2) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
3) You already have two initialized variable named "image" (of type "PIL.Image.Image") and "detected_objects" (of type list[DetectedObject]) - no need to initialize them yourself! 
4) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if specific objects are inside the region of a larger object with the find_overlapping_regions method.
5) It is IMPORTANT to understand that the content of <question></question> will have <regionX> tags (i.e. <region0>, <region1>, etc.). The X index in the tag corresponds to the global variable detected_objects' index. You MAY use the X index (an integer) to get a DetectedObject from the detected_objects array.
6) You MAY want to extract the bounding boxes from the list of DetectedObject (detected_objects) retrieved before using in other functions, as DetectedObject is not a bounding box but a complex class.
7) When the query asks WHICH object is suitable, ANSWER with DetectedObject.description FOR THE TAG ONLY. The <regionX> tags in DetectedObject.description ALWAYS HAVE the '<' and '>' symbols. Please consider that when writing the code.
8) Whenever the query has visual cues (i.e. important to distinguish among similar named objects) - MUST use vqa().
9) The vqa() function returns a STRING. If you want to parse the output as a boolean or numeric value, for example, please do so appropriately using is_similar_text()
10) DO NOT use simple naive math for moderately complex subtasks for the question. When asking questions that require visual understanding better than naive, unintuitive hardcoded method, use vqa().
11) Be more flexiable about positional understanding. For example, if the query is asking the object at the rightmost position, please check the position with a function, and not naive index-based assumptions.
12) ESPECIALLY FOR COUNTING, when checking if some objects are inside another object, use the find_overlapping_regions() function to retrieve the region index (can be used with detected_objects[idx]).
13) When calculating distances between two objects, don't do 2D distance. USE the provided calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject) function from the API above.
14) DetectedObject.description ONLY denotes the type of object - NOT ANY additional relational info (i.e. 'pallet', 'transporter', 'buffer')

Importantly, new methods MUST start with an underscore. As an example, you may define a "_get_material" method. Please ensure you ALWAYS start the name with an underscore.

ONLY build signatures with STANDARD basic typing enforcement / definition for input / output parameters. DO NOT use types that would require import typing for type definition in signature.

Again, output the docstring inside <docstring></docstring> immediately followed by the method signature for the docstring inside <signature></signature>.

DO NOT INCLUDE ``` tags!

Here is the question:
{question}
"""


