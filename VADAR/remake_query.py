import re
def remake_query(query):
    # Generate region IDs based on number of <mask> tokens
    counter = [0]  # Using a list to allow mutation inside replacer

    def replacer(match):
        replacement = f"<region{counter[0]}>"
        counter[0] += 1
        return replacement

    query = re.sub(r'<mask>', replacer, query)
    return query

while True:
    try:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        remade_query = remake_query(query)
        print(f"Remade Query: {remade_query}")
    except Exception as e:
        print(f"An error occurred: {e}")