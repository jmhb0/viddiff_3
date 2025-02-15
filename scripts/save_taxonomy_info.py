"""
python scripts/save_taxonomy_info.py
"""

import ipdb
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("viddiff/VidDiffBench_2", cache_dir=None)
dataset = dataset['test']

def save_differences_to_csv(dataset):
    """
    Collects all differences from the dataset, converts to a DataFrame, and saves to CSV.
    The CSV will be saved in the same directory as this script.
    Removes duplicate entries that have the same action and description.
    """
    # First collect all differences
    differences_by_action = {}
    rows = []
    
    for row in dataset:
        action = row['action']
        differences = json.loads(row['differences_annotated'])
        
        # Add each non-None difference along with its key
        for diff_key, diff_info in differences.items():
            if diff_info is not None:
                # Create a row for the DataFrame
                df_row = {
                    'action': action,
                    'diff_key': diff_key,
                    'name': diff_info['name'],
                    'num_frames': diff_info['num_frames'],
                    'description': diff_info['description'],
                    'query_string': diff_info['query_string']
                }
                rows.append(df_row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Remove duplicates based on action and description
    df = df.drop_duplicates(subset=['action', 'description'])
    
    # Sort by action and diff_key for better readability
    df = df.sort_values(['action', 'diff_key'])
    
    # Save to the same directory as this script
    script_dir = Path(__file__).parent
    output_path = script_dir / 'difference_taxonomy.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Saved differences catalog to: {output_path}")
    # print(f"Total unique differences: {len(df)}")
    # print("\nSummary by action:")
    # print(df.groupby('action').size())

    return df

def save_actions_to_csv(dataset):
    """
    Creates a CSV containing metadata about each action in the dataset.
    Includes action ID, name, description, domain, and source dataset.
    Removes any duplicate actions.
    """
    rows = []
    
    for row in dataset:
        action = row['action']
        # Create a row for the DataFrame
        df_row = {
            'action': action,
            'split': row['split'],
            'action_name': row['action_name'],
            'action_description': row['action_description'],
            'domain': row['domain'],
            'source_dataset': row['source_dataset']
        }
        rows.append(df_row)
    
    # Convert to DataFrame and remove duplicates
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['action'])
    
    # Create a categorical type with custom ordering
    df['split'] = pd.Categorical(df['split'], categories=['easy', 'medium', 'hard'], ordered=True)
    
    # Sort by split (in specified order) and then action
    df = df.sort_values(['split', 'action'])
    
    # Save to the same directory as this script
    script_dir = Path(__file__).parent
    output_path = script_dir / 'actions.csv'
    
    df.to_csv(output_path, index=False)
    
    print(f"Saved actions catalog to: {output_path}")
    print(f"Total unique actions: {len(df)}")
    # print("\nSummary by domain:")
    # print(df.groupby('domain').size())
    ipdb.set_trace()
    
    return df

# Call both functions
save_differences_to_csv(dataset)
save_actions_to_csv(dataset)
ipdb.set_trace()
pass


