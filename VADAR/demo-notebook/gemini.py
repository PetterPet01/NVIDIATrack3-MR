import google.generativeai as genai

genai.configure(api_key='AIzaSyCJHta9VITkW9ZNnTOuqpvPPIrpSSgCqXg')
models = genai.list_models()
for model in models:
    print(model.name)
