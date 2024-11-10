import pandas as pd
import gwenlake

client = gwenlake.Client(api_key="sk_XXX")

list_of_texts = [
    "Do Not Watch This Movie!",
    "This movie is SOOOO funny!!! The acting is WONDERFUL.",
    "Everything about this movie is horrible, from the acting to the editing.",
]
response = client.embeddings.create(input=list_of_texts, model="e5-base-v2")
for item in response.data:
    print(pd.DataFrame(item.embedding))

