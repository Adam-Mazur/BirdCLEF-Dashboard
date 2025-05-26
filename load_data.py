import pandas as pd

# Load the data
train_df = pd.read_csv('data/train.csv')
taxonomy_df = pd.read_csv('data/taxonomy.csv')

merged_df = train_df.merge(
    taxonomy_df[['scientific_name', 'class_name']], 
    on='scientific_name', 
    how='left'
)    

merged_df['author'] = merged_df['author'].astype(str).str.strip()