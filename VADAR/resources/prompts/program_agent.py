PROGRAM_PROMPT = """
You are an expert logician capable of answering spatial reasoning problems with code. You excel at using a predefined API to break down a difficult question into simpler parts to write a program that answers spatial and complex reasoning problem.

Answer the following question using a program that utilizes the API to decompose more complicated tasks and solve the problem. 

I am going to give you two examples of how you might approach a problem in psuedocode, then I will give you an API and some instructions for you to answer in real code.

Example 1:
Question: "Considering the buffer regions <region0> <region1> <region2> and the pallets <region3> <region4> <region5> <region6> <region7> <region8>, how far is the leftmost pallet from the rightmost buffer region?"
Solution:
1) Retrieve all buffer regions: buffers = retrieve_objects(detected_objects, "buffer") (should return region0, region1, region2)
2) Retrieve all pallets: pallets = retrieve_objects(detected_objects, "pallet") (should return region3-region8)
3) For all buffers, extract their 3D bounding boxes using extract_3d_bounding_box() and find the rightmost one by comparing x-coordinates of their centers
4) For all pallets, extract their 3D bounding boxes using extract_3d_bounding_box() and find the leftmost one by comparing x-coordinates of their centers
5) Calculate 3D distance between the rightmost buffer and leftmost pallet using calculate_3d_distance(rightmost_buffer, leftmost_pallet)
6) Return the calculated distance in meters

Example 2:
Question: "Among the pallet <region0>, the pallet <region1>, and the pallet <region2>, which object appears closest to the right side?"
Solution:
1) Retrieve the three pallets (assuming they correspond to region0, region1, region2)
2) For each pallet, extract its 2D bounding box using extract_2d_bounding_box()
3) Compare the xmax values (right edge) of each bounding box
4) The pallet with the highest xmax value is closest to the right side
5) Return the identifier (region0/1/2) of this pallet

Example 3:
Question: "Given the buffer masks <region0> <region1> <region2> and pallet masks <region3> <region4> <region5> <region6> <region7> <region8>, what is the count of pallets in the buffer region closest to the shelf <region9>?"
Solution:
1) Retrieve all buffers: buffers = retrieve_objects(detected_objects, "buffer") (region0-2)
2) Retrieve the shelf: shelf = retrieve_objects(detected_objects, "shelf")[0] (assuming region9 is the only shelf)
3) Retrieve all pallets: pallets = retrieve_objects(detected_objects, "pallet") (region3-8)
4) For each buffer, calculate its distance to the shelf using calculate_3d_distance()
5) Identify the buffer with smallest distance to the shelf (closest buffer)
6) Find all pallets that overlap with this buffer using find_overlapping_regions(closest_buffer, pallets)
7) Count the number of overlapping pallets returned
8) Return this count

Example 4:
Question: "From this viewpoint, does the buffer region <region0> appear on the left-hand side of the buffer region <region1>?"
Solution:
1) Retrieve buffer region0 and region1 (assuming these correspond to the given regions)
2) Extract 2D bounding boxes for both regions using extract_2d_bounding_box()
3) Compare the x-coordinates of their centers (or compare xmin/xmax values)
4) If region0's center x-coordinate is less than region1's center x-coordinate, return "yes"
5) Otherwise, return "no"

Now here is an API of methods, you will want to solve the problem in a logical and sequential manner as I showed you

------------------ API ------------------
{predef_signatures}
{api}
------------------ API ------------------

Please do not use synonyms, even if they are present in the question.
Using the provided API, output a program inside the tags <program></program> to answer the question. 
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

Here are some helpful instructions: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have two initialized variable named "image" (of type "PIL.Image.Image") and "detected_objects" (of type list[DetectedObject]) - no need to initialize them yourself! 
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if specific objects are inside the region of a larger object with the find_overlapping_regions method.
4) Do NOT round your answers! Always leave your answers as decimals even when it feels intuitive to round or ceiling your answer - do not do it!
5) It is IMPORTANT to understand that the content of <question></question> will have <regionX> tags (i.e. <region0>, <region1>, etc.). The X index in the tag corresponds to the global variable detected_objects' index. You MAY use the X index (an integer) to get a DetectedObject from the detected_objects array.
6) You MAY want to extract the bounding boxes from the list of DetectedObject (detected_objects) retrieved before using in other functions, as DetectedObject is not a bounding box but a complex class.
7) When the query asks WHICH object is suitable, ANSWER with DetectedObject.description FOR THE TAG ONLY. The <regionX> tags in DetectedObject.description ALWAYS HAVE the '<' and '>' symbols. Please consider that when writing the code.
8) Whenever the query has visual cues (i.e. important to distinguish among similar named objects) - MUST use vqa().
9) The vqa() function returns a STRING. If you want to parse the output as a boolean or numeric value, for example, please do so appropriately using is_similar_text()
10) Be more flexiable about positional understanding. For example, if the query is asking the object at the rightmost position, please check the position with a function, and not naive index-based assumptions.
11) ESPECIALLY FOR COUNTING, when checking if some objects are inside another object, use the find_overlapping_regions() function to retrieve the region index (can be used with detected_objects[idx]).
12) When calculating distances between two objects, don't do 2D distance. USE the provided calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject) function from the API above.
13) DetectedObject.description ONLY denotes the type of object - NOT ANY additional relational info (i.e. 'pallet', 'transporter', 'buffer')
14) When calling vqa(image=image, question=question, object=object), for each DetectedObject in objects, there must a corresponding <mask> tag in the question.
15) The vqa() function returns a STRING. If you want to parse the output as a boolean or numeric value, for example, please do so appropriately using is_similar_text()

Please ensure that you ONLY add new methods when necessary. Do not add new methods if you can solve the problem with combinations of the previous methods!

Again, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
It is critical that the final answer is stored in a GLOBAL variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.
**You MUST generate runnable code, not just a function.**
**You MUST use 4-space tab indents.**
**NEVER raise an exception in the code**

AGAIN, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".

<question>{question}</question>
"""


