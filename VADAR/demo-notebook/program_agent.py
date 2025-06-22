PROGRAM_PROMPT = """
You are an expert logician capable of answering spatial reasoning problems with code. You excel at using a predefined API to break down a difficult question into simpler parts to write a program that answers spatial and complex reasoning problem.

Answer the following question using a program that utilizes the API to decompose more complicated tasks and solve the problem. 

I am going to give you two examples of how you might approach a problem in psuedocode, then I will give you an API and some instructions for you to answer in real code.

Example 1:
Question: "How many objects have the same color as the metal bowl?"
Solution:
1) Set a counter to 0
2) Find all the bowls (retrieve_objects(detected_objects, 'bowls')).
3) Retrieve the bounding boxes for each DetectedObject of type "bowls" as returned in 1) with extract_2d_bounding_box()
4) If bowls are found, loop through each of the bowls found.
5) For each bowl found, check if the material of this bowl is metal. Store the metal bowl if you find it and break from the loop.
6) Find and store the color of the metal bowl.
7) Find all the objects.
8) For each object O, check if O is the same object as the small bowl i.e. high overlap percentage (find_overlapping_regions(metal_bowl_object, all_objects)). If it is, skip it.
9) For each O you don't skip, check if the color of O is the same as the color of the metal bowl.
10) If it is, increment the counter.
11) When you are done looping, return the counter.

Example 2:
Question: "How many objects of the same height as the mug would you have to stack to achieve an object the same height as the cabinet?"
Solution:
1) Find all the mugs (retrieve_objects(detected_objects, "mug"))
2) Find all the cabinets (retrieve_objects(detected_objects, "cabinet"))
3) Find the 3D dimension of the mug, get the height and store this value.
4) Find the 3D dimension of the cabinet, get the height and store this value.
5) Return the height of the cabinet divided by the height of the mug, do NOT round.

Example 3:
Question: "How many mugs are there in the dishwasher?"
Solution:
1) Find all the mugs (retrieve_objects(detected_objects, "mug"))
2) Retrieve the bounding boxes for each DetectedObject of type "mug" as returned in 1) with extract_2d_bounding_box()
3) Initialize a counter to 0
4) For each mug you found, ask VQA if the mug is in the dishwasher (vqa(image, "Is this mug in the dishwasher?", mug_bbox))
5) If the output is "yes" then increment the counter.
6) Return the counter.

Example 4:
Question: "How many plates are on the table?"
Solution:
1) Find all the plates (retrieve_objects(detected_objects, "plate"))
2) Return the number of plates located.

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
10) DO NOT use simple naive math for moderately complex subtasks for the question. When asking questions that require visual understanding better than naive, unintuitive hardcoded method, use vqa().
11) Be more flexiable about positional understanding. For example, if the query is asking the object at the rightmost position, please check the position with a function, and not naive index-based assumptions.
12) When checking if some objects are inside another object, use the find_overlapping_regions() function to retrieve the region index (can be used with detected_objects[idx]).
13) When calculating distances between two objects, don't do 2D distance. USE the provided calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject) function from the API above.
14) DetectedObject.description ONLY denotes the type of object - NOT ANY additional relational info (i.e. 'pallet', 'transporter', 'buffer')
15) When requested to return the relevant region, you MUST return the DetectedObject.description.

Please ensure that you ONLY add new methods when necessary. Do not add new methods if you can solve the problem with combinations of the previous methods!

You MUST at least build the methods below:
{vqa_functions}

Again, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

AGAIN, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".

<question>{question}</question>
"""


