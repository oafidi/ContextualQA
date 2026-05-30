import json

current_data = []

with open('final/silver_dataset.json', 'r') as f:
    data = json.load(f)
current_data.extend(data)

with open('final/gold_dataset.json', 'r') as f:
    data = json.load(f)

current_data.extend(data)

full_contexts = set()
for item in current_data:
    full_contexts.add(item['context'])

import pandas as pd

df = pd.read_csv('final/goud_preprocessed_V2.csv')

# i need to extract the contexts from the dataframe that are not in the full_contexts set
new_contexts = set(df['context'].tolist()) - full_contexts 

# now i need to create a csv file with the new contexts
new_df = pd.DataFrame(list(new_contexts), columns=['context'])
new_df.to_csv('final/new_contexts.csv', index=False)
print(f"Number of new contexts: {len(new_contexts)}")