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
8) For each object O, check if O is the same object as the small bowl (same_object(image, metal_bowl_bbox, object_bbox)). If it is, skip it.
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

Here are some helpful definitions:
1) 2D distance/size refers to distance/size in pixel space.
2) 3D distance/size refers to distance/size in the real world. 3D size is equal to 2D size times the depth of the object.
3) We define (width, height, length) as the values along the (x, y, z) axis. Width = x axis, height = y axis, length = z axis.
4) "Depth" measures distance from the camera in 3D.

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have two initialized variable named "image" and "detected_objects" - no need to initialize them yourself! 
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.
4) If two objects have the same 2D width, then the object with the largest depth has the largest 3D width.
5) If two objects have the same 2D height, then the object with the largest depth has the largest 3D height.
6) 2D sizes convey the height and width in IMAGE SPACE. To convert to height and width in 3D space, it needs to be multiplied by the depth!
7) Some questions may present hypothesis measurements (e.g. "X is 3.5m wide"), these are hypothesis measurements and should be used ONLY to scale your outputs accordingly.
8) Do NOT round your answers! Always leave your answers as decimals even when it feels intuitive to round or ceiling your answer - do not do it!
9) When a query asks to find all objects in a container just count the number of objects.
10) It is IMPORTANT to understand that the content of <question></question> will have <regionX> tags (i.e. <region0>, <region1>, etc.). The X index in the tag corresponds to the global variable detected_objects' index. You MAY use the X index (an integer) to get a DetectedObject from the detected_objects array.
11) You MAY want to extract the bounding boxes from the list of DetectedObject (detected_objects) retrieved before using in other functions, as DetectedObject is not a bounding box but a complex class.
12) When answering with the global <regionX> tag of an object as the result, use DetectedObject.description.

Again, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

AGAIN, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
You do not need to define a function to answer the question - just write your program in the tags. Assume "image" and "detected_objects" have already been initialized - do not modify it!
<question>{question}</question>
"""

PROGRAM_PROMPT_GQA = """
You are an expert logician capable of answering visual reasoning problems with code.

Answer the following question using a program to solve the problem. 

I am going to give you some examples of how you might approach a problem in psuedocode, then I will give you an API and some instructions for you to answer in real code.

Question: Are there trains or fences in this scene?
Solution:
1) Find all the trains (loc(image, 'trains'))
2) Find all the fences (loc(image, 'fences'))
3) If trains or fences are found, return "yes"
4) Otherwise, return "no"

Question: What color is the curtain that is to the right of the mirror?
Solution:
1) Find the mirror
2) Find all curtains
3) Store the curtain that has an x coordinate greater than the x coordinate of the mirror
4) Ask for the color of the curtain (vqa(image, "what color is this?", curtain_x, curtain_y))
5) Return the color

Question: Is the street light standing behind a truck?
Solution:
1) Find the street light
2) Find all trucks
3) Find the depth of the street light
4) For each truck found, find the depth of the truck
5) If the depth of the street light is greater than the depth of a truck, return "yes"
6) Otherwise, return "no"

Question: Is a cat above the mat?
Solution:
1) Find the mat
2) Find all cats
3) For each cat found, if the y coordinate of the cat is less than the y coordinate of the mat, return "yes"
4) Otherwise, return "no"

Question: Does the mat have the same color as the sky?
Solution:
1) Find the mat
2) Find the sky
3) Ask for the color of the mat (vqa(image, "what color is this?", mat_x, mat_y))
4) Ask for the color of the sky (vqa(image, "what color is this?", sky_x, sky_y))
5) If the color of the mat is the same as the color of the sky, return "yes"
6) Otherwise, return "no"

Question: What do the wetsuit and the sky have in common?
Solution:
1) This question requires a holisitc view of the image, so directly answer the question with vqa. (vqa(image, "what do the wetsuit and the sky have in common?", 0, 0))

Question: Are these animals of different species?
Solution:
1) Look for all animals
2) if there are no animals found, directly answer the question with vqa. (vqa(image, "Are these animals of different species?", 0, 0))
3) Otherwise, find the species of each animal
4) If the species of the animals are different, return "yes"
5) Otherwise, return "no"

Question: Who is wearing a hat?
Solution:
1) Find all people
2) If there are no people found, directly answer the question with vqa. (vqa(image, "Who is wearing a hat?", 0, 0))
3) Otherwise, for each person found, ask if they are wearing a hat (vqa(image, "Is this person wearing a hat?", person_x, person_y))
4) If the person is wearing a hat, ask for their kind (vqa(image, "What kind of person is this?", person_x, person_y))
4) Return the kind of person wearing a hat

Question: Is the purse to the left of the man gold or silver?
Solution:
1) Find the man
2) Find all purses
3) If there are no purses found, directly answer the question with vqa. (vqa(image, "Is the purse to the left of the man gold or silver?", 0, 0))
3) Store the purse that has a x coordinate greater than the x coordinate of the man
4) Since there are only two possible colors, explicity ask for which one of them is true (vqa(image, "Is this gold or silver?", purse_x, purse_y))
5) Return the color of the purse

------------------ API ------------------
{predef_signatures}
{api}
------------------ API ------------------


Using the provided API, output a program inside the tags <program></program> to answer the question. 
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

Here are some helpful tips: 
1) Do not loop through all objects in the scene. Instead, find the objects that satisfy the condition and use them to answer the question.
2) For questions where a holisitc understanding of the image might be necessary, rather than asking about relations between objects, directly answer the question with vqa.
3) If the question asks to choose between two choices, ensure that any vqa calls ask to choose between the two options for the object being referred to, rather than for the generic attribute of the object.
4) Never return "none" or "unknown", if you cannot find an certain object, directly answer the question with vqa.
5) For questions that ask about a person like "Who is wearing a skirt?", ask about the kind of person rather than the person themselves.
You already have an initialized variable named "image" - no need to initialize it yourself! 
5) Do not define new methods here, simply solve the problem using the existing methods provided in the API.

Again, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

AGAIN, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
You do not need to define a function to answer the question - just write your program in the tags. Assume "image" has already been initialized - do not modify it!
<question>{question}</question>
"""

PROGRAM_PROMPT_CLEVR = """
You are an expert logician capable of answering spatial reasoning problems with code. You excel at using a predefined API to break down a difficult question into simpler parts to write a program that answers spatial and complex reasoning problem.

Answer the following question using a program that utilizes the API to decompose more complicated tasks and solve the problem. 
Available sizes are {{small, large}}, available shapes are {{square, sphere, cylinder}}, available material types are {{rubber, metal}}, available colors are {{gray, blue, brown, yellow, red, green, purple, cyan}}.

The question may feature attributes that are outside of the available ones I specified above. If that's the case, please replace them to the most appropriate one from the attributes above.

I am going to give you an example of how you might approach a problem in psuedocode, then I will give you an API and some instructions for you to answer in real code.

Example:

Question: "What is the shape of the matte object in front of the red cylinder?"
Solution:
1) Find all the cylinders (loc(image, 'cylinders'))
2) If cylinders are found, loop through each of the cylinders found
3) For each cylinder found, check if the color of this cylinder is red. Store the red cylinder if you find it and break from the loop.
4) Find all the objects.
5) For each object, check if the object is rubber (matte is not in the available attributes, so we replace it with rubber)
3) For each rubber object O you found, check if the depth of O is less than the depth of the red cylinder
4) If that is true, return the shape of that object

Now here is an API of methods, you will want to solve the problem in a logical and sequential manner as I showed you

------------------ API ------------------
{predef_signatures}
{api}
------------------ API ------------------

Please do not use synonyms, even if they are present in the question.
Using the provided API, output a program inside the tags <program></program> to answer the question. 
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have an initialized variable named "image" - no need to initialize it yourself! 3) Do not define new methods here, simply solve the problem using the existing methods.
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.

Again, available sizes are {{small, large}}, available shapes are {{square, sphere, cylinder}}, available material types are {{rubber, metal}}, available colors are {{gray, blue, brown, yellow, red, green, purple, cyan}}.

Again, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

AGAIN, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
You do not need to define a function to answer the question - just write your program in the tags. Assume "image" has already been initialized - do not modify it!

DO NOT INCLUDE ``` tags!
<question>{question}</question>
"""

PROGRAM_PROMPT_VSI = """
You are an expert logician capable of answering spatial reasoning problems with code. You excel at using a predefined API to break down a difficult question into simpler parts to write a program that answers spatial and complex reasoning problem.

Answer the following question using a program that utilizes the API to decompose more complicated tasks and solve the problem. 

I am going to give you two examples of how you might approach a problem in psuedocode, then I will give you an API and some instructions for you to answer in real code.

Example 1:

Question: "What is the shape of the red object in front of the blue pillow?"
Solution:
1) Find all the pillows (loc(image, 'pillow')).
2) If pillows are found, loop through each of the pillows found.
3) For each pillow found, check if the color of this pillow is blue. Store the blue pillow if you find it and break from the loop.
4) Find all the objects.
5) For each object, check if the object is red.
6) For each red object O you found, check if the depth of O is less than the depth of the blue pillow.
7) If that is true, return the shape of that object.

Example 2:
Question: "How many objects have the same color as the metal bowl?"
Solution:
1) Set a counter to 0
2) Find all the bowls (loc(image, 'bowls')).
3) If bowls are found, loop through each of the bowls found.
4) For each bowl found, check if the material of this bowl is metal. Store the metal bowl if you find it and break from the loop.
5) Find and store the color of the metal bowl.
6) Find all the objects.
7) For each object O, check if O is the same object as the small bowl (same_object(image, metal_bowl_x, metal_bowl_y, object_x, object_y)). If it is, skip it.
8) For each O you don't skip, check if the color of O is the same as the color of the metal bowl.
9) If it is, increment the counter.
10) When you are done looping, return the counter.

Now here is an API of methods, you will want to solve the problem in a logical and sequential manner as I showed you

------------------ API ------------------
{predef_signatures}
{api}
------------------ API ------------------

Please do not use synonyms, even if they are present in the question.
Using the provided API, output a program inside the tags <program></program> to answer the question. 
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

Here are some helpful definitions:
1) 2D distance/size refers to distance/size in pixel space.
2) 3D distance/size refers to distance/size in the real world. 3D size is equal to 2D size times the depth of the object.
3) "On" is defined as the closest object ABOVE another object. Only use this definition for "on".
4) "Next to" is defined as the closest object.
5) Width is the same as length.
6) "Depth" measures distance from the camera in 3D in meters.

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have an initialized variable named "image" - no need to initialize it yourself! 
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.
4) Do not assume that the objects you see in these questions rae all of the objects you will see, keep the methods general.
5) If two objects have the same 2D width, then the object with the largest depth has the largest 3D width.
6) If two objects have the same 2D height, then the object with the largest depth has the largest 3D height.
7) 2D sizes convey the height and width in IMAGE SPACE. To obtain absolute measures of height and width in 3D space, you need to multiply the 2D size by the depth and divide by the focal length!
8) If you are given a reference size, scale your output predicted size accordingly!
9) PAY ATTENTION TO UNITS. Do not assume things will be in either centimeters or meters. The question will tell you which unit to use. Scale your outputs accordingly.
10) A pixel (x,y) on the image can be converted to a 3D point (Z * (x - px)/f, Z * (y - py)/f, Z) whrere f is the focal length, Z is the depth, and (px, py) is the principal point. You can assume the principal point is at the center of the image (px = W/2, py = H/2). You can access the (width, height) of the image by doing image.size

Again, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

AGAIN, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
You do not need to define a function to answer the question - just write your program in the tags. Assume "image" has already been initialized - do not modify it!

DO NOT INCLUDE ``` tags!

<question>{question}</question>
"""
