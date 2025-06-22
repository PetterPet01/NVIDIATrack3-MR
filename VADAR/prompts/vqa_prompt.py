VQA_PROMPT_CLEVR = """You will be shown an image with a bounding box and asked to answer a question based on the object inside the bounding box.
Available sizes are {{small, large}}, available shapes are {{cube, sphere, cylinder}}, available material types are {{rubber, metal}}, available colors are {{gray, blue, brown, yellow, red, green, purple, cyan}}. Please DO NOT answer with attributes other than those that I specified!
If you think the right answer isn't one of the available attributes, choose the available attribute that is most similar!
Answer this question using the attributes as reference based on the object in the bounding box and put your answer in between <answer></answer> tags: {question}"""

VQA_PROMPT_GQA = """
You will be shown an image with a bounding box and asked to answer a question based on the object inside or around the bounding box.
Your answer should be a single word.
The answer should be simple and generic.
If the answer is not generic I will kill you.
If asked about people, do not just say "person".
Never respond with "unknown" or "none", always give a plausible answer.
Put your answer in between <answer></answer> tags: 
Question:{question}"""

VQA_PROMPT_GQA_HOLISTIC = """
Answer this question. 
Your answer should be a single word.
If the answer is based on an object, the answer should be simple and generic.
If the answer is not generic I will kill you.
If asked about people, do not just say "person".
Never respond with "unknown" or "none", always give a plausible answer.
Put your answer in between <answer></answer> tags: 
Question:{question}"""

VQA_PROMPT = """You will be shown an image with a red bounding box and asked to answer a question based on the object inside the bounding box. Please only answer with regards to the object IN the bounding box. Answer this question with one word based on the object in the bounding box and put your answer in between <answer></answer> tags: {question}"""
