import os
import re
import csv
#Get short ID for subject with age <= 15

# Folder containing your .1D files
data_dir = '/home/celery/Documents/Research/dataset/Outputs/cpac/filt_global/rois_ho/'

# Output file path
output_txt = '/home/celery/Documents/Research/dataset/under15_short_IDs.txt'

# Path to phenotypic file
phenotype_csv = '/home/celery/Documents/Research/dataset/Phenotypic_V1_0b_preprocessed1.csv'

# Regex pattern to extract 7-digit subject ID
pattern = re.compile(r'_(\d{7})_rois_ho\.1D$')

# Load age info
id_to_age = {}
with open(phenotype_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sub_id = row['SUB_ID'].lstrip('0')  # remove leading zeros
        try:
            id_to_age[sub_id] = float(row['AGE_AT_SCAN'])
        except:
            continue

# Extract IDs
subject_ids = []
for filename in os.listdir(data_dir):
    match = pattern.search(filename)
    if match:
        subject_id = str(int(match.group(1)))  # convert to int → removes leading zeros
        age = id_to_age.get(subject_id)
        if age is not None and age <= 15:
            subject_ids.append(subject_id)

# Save to .txt file
with open(output_txt, 'w') as f:
    for sub_id in subject_ids:
        f.write(sub_id + '\n')

print("Saved", len(subject_ids), "subject IDs (no leading zeros, age ≤ 15) to:", output_txt)

#Get full ID for subject with age <= 15

# File with valid short IDs 
short_id_txt = '/home/celery/Documents/Research/dataset/under15_short_IDs.txt'

# Output file
output_full_id_txt = '/home/celery/Documents/Research/dataset/under15_full_IDs.txt'

# Load short IDs (with zero-padding preserved)
with open(short_id_txt) as f:
    short_ids = set(line.strip().zfill(7) for line in f if line.strip())

# New pattern: match any prefix, then a 7-digit ID, then _rois_ho.1D
pattern = re.compile(r'^(.*?_(\d{7}))_rois_ho\.1D$')

full_ids = []

for filename in os.listdir(data_dir):
    match = pattern.match(filename)
    if match:
        full_id = match.group(1)        # e.g. Leuven_1_0050689
        padded_id = match.group(2)      # e.g. 0050689
        if padded_id in short_ids:
            full_ids.append(full_id)

# Save results
with open(output_full_id_txt, 'w') as f:
    for fid in full_ids:
        f.write(fid + '\n')

print("Saved", len(full_ids), "full IDs to:", output_full_id_txt)
