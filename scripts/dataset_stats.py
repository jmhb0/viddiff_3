"""
python -m ipdb scripts/dataset_stats.py
"""
import datasets
import ipdb
import json

dataset = datasets.load_dataset("viddiff/VidDiffBench_2")['test']
domains = set(dataset['domain'])

# video count statistics
domain_counts = {domain: 0 for domain in domains}
print(f"Number of samples total: {len(dataset)}")
for row in dataset:
    domain_counts[row['domain']] += 1
for domain, count in domain_counts.items():
    print(f"  {domain}: {count}") 
print()


# difference statistics
num_diffs = 0
diff_strings = set()
domains_diffs = {domain:0 for domain in domains}
for row in dataset:
    diffs = json.loads(row['differences_annotated'])
    diffs = {k:v for k,v in diffs.items() if v is not None}
    num_diffs += len(diffs)
    domains_diffs[row['domain']] += len(diffs)
    for k, v in diffs.items():
        if v is None: 
            continue
        diff_string = row['action'] + v['description']
        diff_strings.add(diff_string)
print(f"Number of diffs: {num_diffs}")
for domain, num_diffs in domains_diffs.items():
    print(f"  {domain}: {num_diffs}")
print("Number of unique diff strings: ", len(diff_strings))


# difference type ratios
print("\nDifference type ratios by split:")
splits = ['easy', 'medium', 'hard']
split_stats = {split: {'a': 0, 'b': 0, 'c': 0} for split in splits}
total_stats = {'a': 0, 'b': 0, 'c': 0}

for row in dataset:
    diffs_gt = json.loads(row['differences_gt'])
    split = row['split']
    for diff_type in diffs_gt.values():
        if diff_type is not None:
            split_stats[split][diff_type] += 1
            total_stats[diff_type] += 1

# Print ratios for each split
for split in splits:
    counts = split_stats[split]
    total = sum(counts.values())
    if total > 0:
        ab_ratio = counts['a'] / counts['b'] if counts['b'] > 0 else float('inf')
        print(f"\n{split.capitalize()} split:")
        print(f"  A:B ratio = {ab_ratio:.2f}")
        print(f"  A:B:C distribution = {counts['a']/total:.2f} : {counts['b']/total:.2f} : {counts['c']/total:.2f}")

# Print overall ratios
total = sum(total_stats.values())
if total > 0:
    ab_ratio = total_stats['a'] / total_stats['b'] if total_stats['b'] > 0 else float('inf')
    print(f"\nOverall:")
    print(f"  A:B ratio = {ab_ratio:.2f}")
    print(f"  A:B:C distribution = {total_stats['a']/total:.2f} : {total_stats['b']/total:.2f} : {total_stats['c']/total:.2f}")

ipdb.set_trace()
pass
