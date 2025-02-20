"""
python -m ipdb scripts/update_dataset.py 
"""
import ipdb
import pdb
import os
import numpy as np
import json
import re
from PIL import Image
from pathlib import Path
from datasets import load_dataset
import decord
from tqdm import tqdm
import logging
import hashlib
from datasets import DatasetDict, Dataset
import pandas as pd

dataset = load_dataset("viddiff/VidDiffBench", cache_dir=None)
dataset = dataset['test']

# now we make changes to the dataset
# Create a new column by mapping the existing split values
dataset = dataset.map(lambda x: {'domain': x['split']})
def rename_splits(dataset):
    """ Make the 'splits' correspond to easy/medium/hard and put the old 'splits' word into 'domain'"""
    def _get_difficulty_splits(dataset):
        """ 
        Create a new column called 'split_difficulty' by looking it up. 
        """
        with open("data/lookup_action_to_split.json", "r") as fp:
            lookup_action_to_split = json.load(fp)

        def add_split_difficulty(example):
            example['split_difficulty'] = lookup_action_to_split[example['action']]
            return example

        dataset = dataset.map(add_split_difficulty)
        return dataset

    dataset = _get_difficulty_splits(dataset)
    # Rename column 'split_difficulty' to 'split', overwriting the existing 'split' column
    dataset = dataset.remove_columns('split')
    dataset = dataset.rename_column('split_difficulty', 'split')


    return dataset

def _print_surgery_and_music_diffs(dataset):
    """ used this to just inspect the current diff names so I can update them"""
    for action in ["surgery_0", "surgery_1", "surgery_2", "music_0", "music_1", "diving_0"]:
        print(action)
        samples_this_action = dataset.filter(lambda x: x['action'] == action)
        diffs = json.loads(samples_this_action[0]['differences_annotated'])
        diffs = {k: v['description'] for k, v in diffs.items() if v is not None}
        for k, v in diffs.items():
            print(k, v)
        print()

def filter_ambiguous_diffs(dataset):
    # next change: filter the diffs I promised to filter
    # there are 2 from surgery and 1 in music
    filter_diffs = {
        "surgery_0": ["The movements in Video A are more efficient than in Video B.",
                      "The movements in Video A are more precise than in Video B."],
        "surgery_1": [
                      "The movement is more fluid in Video A than in Video B.",
                      ],
        "surgery_3" : ["The passage of the needle between two hands is more fluid in Video A than in Video B.",                      
                        "The thread is more efficiently managed in Video A than in Video B.",
                       ],
        "music_0": ["Rhythmic consistency is better maintained in Video A than in Video B.",
        ]
        
    }
    def _filter_row(row):
        if row['action'] in filter_diffs:
            diffs = json.loads(row['differences_annotated'])
            labels = json.loads(row['differences_gt'])
            
            for k, v in diffs.items():
                if v is not None and v['description'] in filter_diffs[row['action']]:
                    diffs[k] = None
                    labels[k] = None
            
            row['differences_annotated'] = json.dumps(diffs)
            row['differences_gt'] = json.dumps(labels)
        return row

    return dataset.map(_filter_row)

def rename_videoa_videob(dataset):
    """ do a more standard naming format  """
    rewrite_description = {
        # surgery_0
        "The movements in Videos A are faster than in Video B." : "The movements are faster.",
        "The tension on the suturing material and the tissue is better controlled in Video A than in Video B." : "The tension on the suturing material and the tissue is better controlled",
        "The surgeon in Video A stops more often to plan next steps than the surgeon in Video B." : "The surgeon stops more often to plan next steps",
        "Both graspers are used efficiently." : "Both graspers are used efficiently.",
        "More errors are corrected in Video A than in Video B." : "More errors are corrected",
        "The suturing thread tangles." : "The suturing thread tangles.",
        "The tube in Video A moves more than in Video B." : "The tube moves more",
        # surgery_1
        "The grasper in Video A is more quickly positioned on the needle than in Video B." : "The grasper is more quickly positioned on the needle",
        "The grasper grasps the needle approximately 2/3 from the needle tip. The needle is grasped more precisely in Video A than in Video B." : "The grasper is more precisely positioned to the correct position, which is approximately 2/3 from the needle tip",
        "The suturing speed is higher in Video A than in Video B" : "The suturing speed is higher",
        "The dot is more accurately hit in Video A than in Video B" : "The dot is more accurately hit",
        "The needle is inserted in the fabric more perpendicular to the incision." : "The needle is inserted in the fabric more perpendicular to the incision.",
        "The force is applied in a more radial way in Video A than in Video B." : "The force is applied in a more radial way",
        "The left graser supports the right grasper, by pressing down on the tissue." : "The left graser supports the right grasper, by pressing down on the tissue.",
        "The instrument tips are never out of view (occluded by instruments, or out of frame)" : "The instrument tips are never out of view (occluded by instruments, or out of frame)",
        "The instrument applies more force to the tissue and needle in Video A than in Video B" : "The instrument applies more force to the tissue and needle",
        "The tension on the suturing thread is lower in Video A than in Video B." : "The tension on the suturing thread is lower",
            # surgery_2
        "The needle is grasped closer to the tip in Video A than in video B." : "The needle is grasped closer to the tip",
        "The movement of the needle through the hoop is more radial in Video A than in Video B." : "The movement of the needle through the hoop is more radial",
        "The passage of the needle between two hands is more fluid in Video A than in Video B." : "The passage of the needle between two hands is more fluid",
        "The instrument tips are never out of view (occluded by instruments, or out of frame)" : "The instrument tips are never out of view (occluded by instruments, or out of frame)",
        "The force on the target is lower in Video A than in Video B" : "The force on the target is lower",
        "The second grasper is used to stabalize the target." : "The second grasper is used to stabalize the target.",
        "The thread is more efficiently managed in Video A than in Video B." : "The thread is more efficiently managed",
        "The number of movements to arrange the needle before threading is lower in Video A than in Video B." : "The number of movements to arrange the needle before threading is lower",
        # music_0
        "There are more wrong note corrections in Video A than in Video B." : "There are more wrong note corrections",
        "The wrist should be straight and not dipped or raised, facilitating fluid motion and avoiding strain. The wrist position is more appropriate in Video A than in Video B." : "The wrist is more straight and not dipped or raised",
        "The speed of playing is higher in Video A than in Video B." : "The speed of playing is faster",
        "The smoothness of thumb crossing is more evident in Video A than in Video B." : "The thumb crosses more smoothly",
        "Forearm movement is more controlled and minimal in Video A than in Video B." : "There is less forearm movement",
        "The body is closer to the piano in Video A than in Video B." : "The body is closer to the piano",
        # music_1 
        "Fingers should press strings at the center of the frets, avoiding the metal fret bars for clear sound production. Video A shows more accurate finger placement on the fretboard than Video B." : "Finger palcement on the strings is closer to the center of the frets, avoiding the metal fret bars",
        "Smooth transitions between strings with minimal disruption to the rhythm or tempo. Transitions in Video A are smoother than in Video B." : "Smoother transitions between strings with minimal disruption",
        "The player uses a plectrum." : "The player uses a plectrum",
        "The left fingers in Video A are more curved / less collapsed than in video B." : "The left fingers are more curved and less collapsed",
        "Only one finger of the left hands rests on a string at a time." : "Only one finger of the left hands rests on a string at a time",
        "The unused left finger tips in Video A stay closer to the board than in video B." : "The unused left finger tips stay closer to the board",
        "Guiatrist uses finger vibrato." : "The guiatrist uses finger vibrato",
        # diving_0
        "Diver enters the water at an angle closer to 90 degrees in video A than in video B." : "Diver enters the water at an angle closer to 90 degrees",
        "The size and volume of the splash created upon entry is larger for video A than video B." : "The size and volume of the splash created upon entry is larger",
        "Duration from jump off the board to water entry in longer in video A than in video B." : "Duration from jump off the board to water entry is longer",
        "Speed at which divers rotate during the dive in larger in video A than in video B." : "Speed at which divers rotate during the dive is faster",
        "Diver faces the water at jump off." : "Diver faces the water at jump off",
        "Diver rotates forward relative to themselves." : "Diver rotates forward relative to themselves",
        "Diver's body is more straight in video A than in video B." : "Diver's body is more straight",
        }
    
    def _rewrite_row(row):
        if row['action'] in ['surgery_0', 'surgery_1', 'surgery_2', 'music_0', 'music_1', 'diving_0']:
            diffs = json.loads(row['differences_annotated'])
            
            for k, v in diffs.items():
                if v is not None:
                    assert v['description'] in rewrite_description, f"{v['description']}"
                    v['description'] = rewrite_description[v['description']] 
            
            row['differences_annotated'] = json.dumps(diffs)
        return row
    
    return dataset.map(_rewrite_row)


def count_difficulty_by_split(dataset):
    # Initialize counters for each split
    counts = {}
    
    for row in dataset:
        split = row['split']
        gt_labels = json.loads(row['differences_gt'])
        
        if split not in counts:
            counts[split] = {'a': 0, 'b': 0, 'c': 0}
        
        # Count each label in this row's gt_labels
        for label in gt_labels.values():
            if label is not None:  # Skip None values
                counts[split][label.lower()] += 1
    
    # Print the results
    for split, diff_counts in counts.items():
        print(f"\nSplit: {split}")
        print(f"  a: {diff_counts['a']}")
        print(f"  b: {diff_counts['b']}")
        print(f"  c: {diff_counts['c']}")
        total = sum(diff_counts.values())
        print(f"  Total: {total}")
        
        # Calculate percentage of 'a' out of 'a' and 'b'
        ab_total = diff_counts['a'] + diff_counts['b']
        if ab_total > 0:
            a_percentage = (diff_counts['a'] / ab_total) * 100
            print(f"  Percentage of 'a' among 'a' and 'b': {a_percentage:.1f}%")


def count_ab_ratio(dataset, split='easy'):
    counts = {'a': 0, 'b': 0, 'c': 0}
    split_samples = dataset.filter(lambda x: x['split'] == split)
    
    for row in split_samples:
        gt_labels = json.loads(row['differences_gt'])
        for label in gt_labels.values():
            if label is not None:
                counts[label.lower()] += 1
    
    ab_total = counts['a'] + counts['b']
    a_percentage = (counts['a'] / ab_total * 100) if ab_total > 0 else 0
    return counts, a_percentage

def balance_easy_split(dataset, n_rows=50):
    # Print initial ratio
    counts, a_percentage = count_ab_ratio(dataset)
    print(f"Initial ratio in easy split:")
    print(f"  a: {counts['a']}, b: {counts['b']}")
    print(f"  Percentage of 'a': {a_percentage:.1f}%")
    
    def _flip_row(row):
        # Count a's and b's in this row
        gt_labels = json.loads(row['differences_gt'])
        row_counts = {'a': 0, 'b': 0}
        for label in gt_labels.values():
            if label in ['a', 'b']:
                row_counts[label] += 1
        
        # If more b's than a's, flip the row
        if row_counts['b'] > row_counts['a']:
            # Flip videos order
            videos = json.loads(row['videos'])
            row['videos'] = json.dumps(videos[::-1])
            
            # Swap thumbnails
            vid0_thumb = row['vid0_thumbnail']
            row['vid0_thumbnail'] = row['vid1_thumbnail']
            row['vid1_thumbnail'] = vid0_thumb
            
            # Flip a's and b's in gt_labels
            for k, v in gt_labels.items():
                if v == 'a':
                    gt_labels[k] = 'b'
                elif v == 'b':
                    gt_labels[k] = 'a'
            row['differences_gt'] = json.dumps(gt_labels)
            
            return True
        return False
    
    # Apply flips to first n_rows of easy split
    flipped_count = 0
    new_dataset = []
    
    for i, row in enumerate(dataset):
        new_row = dict(row)
        if row['split'] == 'easy' and flipped_count < n_rows:
            if _flip_row(new_row):
                flipped_count += 1
        new_dataset.append(new_row)
    
    # Convert back to Dataset
    new_dataset = Dataset.from_list(new_dataset)
    
    # Print final ratio
    counts, a_percentage = count_ab_ratio(new_dataset)
    print(f"\nFinal ratio in easy split after flipping {flipped_count} rows:")
    print(f"  a: {counts['a']}, b: {counts['b']}")
    print(f"  Percentage of 'a': {a_percentage:.1f}%")
    
    return new_dataset

def remove_fps_from_videos(dataset):
    def _process_row(row):
        videos = json.loads(row['videos'])
        # Assert both videos have 'fps' key
        assert 'fps' in videos[0], f"First video missing 'fps' key in row"
        assert 'fps' in videos[1], f"Second video missing 'fps' key in row"
        
        # Remove 'fps' from both video dicts
        videos[0].pop('fps')
        videos[1].pop('fps')
        
        # Save back to row
        row['videos'] = json.dumps(videos)
        return row
    
    return dataset.map(_process_row)

def set_n_differences(dataset):
    """
    Creates a new column 'n_differences' that is 1.5x the number of non-None differences,
    rounded up to the nearest integer.
    """
    def _process_row(row):
        # Load the differences from the JSON string
        gt_diffs = json.loads(row['differences_gt'])
        
        # Count non-None differences
        n_actual_diffs = sum(1 for diff in gt_diffs.values() if diff is not None)
        
        # Calculate target number (1.5x rounded up)
        row['n_differences_open_prediction'] = int(np.ceil(n_actual_diffs * 1.5))
        return row
    
    return dataset.map(_process_row)

def check_description_matches_query_string(dataset):
    """
    For each row, check if the descriptions in differences_annotated match their query_strings.
    Prints a report showing matches/total for each sample.
    """
    for row in dataset:
        diffs = json.loads(row['differences_annotated'])
        
        # Count matches and total non-None differences
        matches = 0
        total = 0
        
        for diff in diffs.values():
            if diff is not None:
                total += 1
                if diff['description'] == diff['query_string']:
                    matches += 1
        
        # Print results if there are any differences
        if total > 0:
            print(f"sample {row['sample_key']}: {matches}/{total} matches")
            
            # Optionally print mismatches for debugging
            if matches < total:
                print("Mismatches:")
                for diff in diffs.values():
                    if diff is not None and diff['description'] != diff['query_string']:
                        print(f"  Description: {diff['description']}")
                        print(f"  Query string: {diff['query_string']}\n")

def find_only_c_samples(dataset):
    """
    Filter out rows where differences_gt only contains 'c' or null values,
    with no 'a' or 'b' values.
    Returns the filtered dataset.
    """
    def has_ab_labels(row):
        gt_labels = json.loads(row['differences_gt'])
        labels = set(label for label in gt_labels.values() if label is not None)
        # Keep row if it has at least one 'a' or 'b' label
        return not labels.issubset({'c'})
    
    filtered_dataset = dataset.filter(has_ab_labels)
    
    # Print info about removed samples
    n_removed = len(dataset) - len(filtered_dataset)
    print(f"Removed {n_removed} samples that only had 'c' or null labels")
    
    return filtered_dataset

def collect_differences_by_action(dataset):
    """
    Collects all differences from the dataset organized by action.
    Returns a dictionary where keys are actions and values are lists of difference dictionaries.
    """
    differences_by_action = {}
    
    for row in dataset:
        action = row['action']
        differences = json.loads(row['differences_annotated'])
        
        if action not in differences_by_action:
            differences_by_action[action] = []
            
        # Add each non-None difference along with its key
        for diff_key, diff_info in differences.items():
            if diff_info is not None:
                # Add the key to the difference info
                diff_info['diff_key'] = diff_key
                differences_by_action[action].append(diff_info)
    
    # Print summary
    for action, diffs in differences_by_action.items():
        print(f"\n{action}: {len(diffs)} differences")
        for diff in diffs:
            print(f"  - {diff['description']}")
    
    return differences_by_action

dataset = rename_splits(dataset)
dataset = filter_ambiguous_diffs(dataset)
dataset = rename_videoa_videob(dataset)
dataset = balance_easy_split(dataset, n_rows=7)
dataset = remove_fps_from_videos(dataset)
dataset = set_n_differences(dataset)
# count_difficulty_by_split(dataset)
# check_description_matches_query_string(dataset)
print("dataset before filtering", len(dataset))
dataset = find_only_c_samples(dataset)
print("dataset after filtering", len(dataset))

ipdb.set_trace()
pass


# Push to hub
if 1:
    dataset_dict = DatasetDict({"test": dataset})
    dataset_dict.push_to_hub("viddiff/VidDiffBench_2", private=False)
ipdb.set_trace()
pass
