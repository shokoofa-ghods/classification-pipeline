import pydicom
import os
import numpy as np
import cv2
import pydicom
from glob import glob
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# make csv files from directory addresses
def process_label_df(df):
    df['path'] = df.apply(lambda row: '/'.join([row['home_folder_name'], str(row['folder_name']), str(row['file_name']).split('.')[0]]), axis=1)
    return df

def create_label_df(amy, non_amy):
    non_amy['home_folder_name'] = 'Tahseen'
    processed_amy = process_label_df(amy)
    processed_non_amy = process_label_df(non_amy)
    label_df = pd.concat([processed_amy, processed_non_amy], axis=0).reset_index()
    return label_df

def find_existing_file_paths(original_address):
    paths = glob(os.path.join(original_address,'*/**', '*'))
    return [path.split(f'{original_address}')[-1] for path in paths]

def find_all_paths(addresses):
    all_path = []
    for address in addresses:
        all_path.extend(find_existing_file_paths(address))
    return all_path

# this is to adjust with missing labels in amyloid data 
def filter_basedOn_labels(all_paths, label_df):
    new_all_paths = []
    filtered = list(label_df[~label_df['label'].isna()]['path'])
    for path in all_paths:
        if path in filtered:
            new_all_paths.append(path)
    return new_all_paths

def create_csv(paths, parent_dir, label_df, amy_folders):
    data_dict = {'path':[], 'frame':[], 'label':[], 'disease':[], 'sample_spacing':[], 'start':[], 'end':[]}
    disease = 0
    label = None
    original_path = ''
    
    for path in paths:
        if path.split('/')[0] not in amy_folders:
            original_path = parent_dir[0]
        else:
            original_path = parent_dir[1]
        imgs = glob(os.path.join(original_path, path, '*'))

        for v in imgs:
            if v.endswith('.npy'):
                imgs.remove(v)
        
        sorted_imgs = sorted([img.split('_')[-1].split('.png')[0] for img in imgs])
        if len(imgs)<10:
            continue
        
        start, end = int(sorted_imgs[0]), int(sorted_imgs[-1])
        selected_frames, retstep = np.linspace(start, end, num= 10, retstep=True)

        if path.split('/')[-3] in amy_folders:
            disease = 1
        
        label = label_df[label_df['path'] == path]['label'].values[0]

        for frame in selected_frames:
            data_dict['path'].append(path)
            data_dict['frame'].append(int(frame))
            data_dict['label'].append(label)
            data_dict['disease'].append(disease)
            data_dict['sample_spacing'].append(retstep)
            data_dict['start'].append(start)
            data_dict['end'].append(end)

    return pd.DataFrame(data_dict)

# keep standard views 
def filter_views(df, items):
    def filter_item(x):
        x = x.rstrip()
        parts = x.split('-')
        filtered = [p for p in parts if p.lower() not in items]
        return '-'.join(filtered)
    df['label'] = df['label'].apply(filter_item)
    df = df[df['label'] != 'Other']
    df = df[df['label'] != 'CW']
    df = df[df['label'] != 'Other']
    df = df[df['label'] != 'CW']
    df = df[df['label'] != '4ch'] #constrast blood pool, different image from apical 4ch
    df = df[df['label'] != '3ch']
    df = df[df['label'] != '2ch']
    df = df[df['label'] != 'PSAX-ves-base']


    return df


# create stratified train/val/test splits
def encoding(patient_df):
    mlb = MultiLabelBinarizer()     # One-hot encode the multi-label views
    view_matrix = mlb.fit_transform(patient_df['label'])

    stratify_features = np.hstack([view_matrix, patient_df[['disease']].values]) # Combine views and health status into stratification features
    return stratify_features


def group_patient(df): # Group view labels per patient
    patient_df = df.groupby('patient_id').agg({
        'label': lambda x: list(set(x)),  # Unique views per patient
        'disease': 'first'  # Assumes consistent health status per patient
    }).reset_index()
    return patient_df

def create_split(patient_df, stratify_features, df):
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_val_idx, test_idx = next(mskf.split(patient_df['patient_id'], stratify_features)) # Split into train+val vs test (say 80-20 first)

    train_idx, val_idx = next( # Apply another split to train_val for train vs val (say 75-25)
        mskf.split(
            patient_df.iloc[train_val_idx]['patient_id'],
            stratify_features[train_val_idx]
        )
    )

    train_patients = patient_df.iloc[train_val_idx].iloc[train_idx]['patient_id'].tolist()     # Get patient IDs
    val_patients = patient_df.iloc[train_val_idx].iloc[val_idx]['patient_id'].tolist()
    test_patients = patient_df.iloc[test_idx]['patient_id'].tolist()

    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]

    train_df.drop(['patient_id'], axis=1, inplace=True)
    val_df.drop(['patient_id'], axis=1, inplace=True)
    test_df.drop(['patient_id'], axis=1, inplace=True)

    
    return train_df, val_df, test_df

def make_csv(df , df_name = 'df', dest_dir=''):
    df.to_csv(os.path.join(dest_dir,f'{df_name}.csv'), index=False)
    print(f"saved {df_name} csv.")


# restructure directories based on the train/val/test splits
def copy_files_by_split(train_df, val_df, test_df,
                        amy_folders,
                        source_root,
                        filepath_col = 'path',
                        output_root='split_data'):
    
    split_map = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    for split, df in split_map.items():
        print(f"\n Copying {len(df)} files to '{split}' folder...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Copying {split}'):
            rel_path = row[filepath_col]  # relative path to the file
            if rel_path.split('/')[0] in amy_folders:
                src_path = os.path.join(source_root[1], rel_path)
            else:
                src_path = os.path.join(source_root[0], rel_path)
                rel_path = '/'.join(rel_path.split('/')[1:]) 
            dest_path = os.path.join(source_root[-1], output_root, split, rel_path)

            if not os.path.exists(src_path):
                print(f"⚠️ Source folder not found: {src_path}")
                continue

            os.makedirs(dest_path, exist_ok=True) # Create destination folder if it doesn't exist

            for filename in os.listdir(src_path): # Copy all files from src_folder to dest_folder
                src_file = os.path.join(src_path, filename)
                dest_file = os.path.join(dest_path, filename)

                if os.path.isfile(src_file):
                    try:
                        shutil.copy2(src_file, dest_file)
                    except Exception as e:
                        print(f" Error copying {src_file}: {e}")

        print(f" Done copying for '{split}'.")


# info
amy_folders = ['TTE', 'US_guided_biopsy', 'cardiac_stress_study']
addresses = pd.read_csv('data_address.csv')
amy_labels = pd.read_csv('Amyloid_echo_view_labels - Sheet1.csv')
non_amy_labels = pd.read_csv('NEW_Tahseen_echo_view_labels - Sheet1.csv')
parent_dir = list(addresses['path'])

# create df using directories
label_df = create_label_df(amy_labels, non_amy_labels)
all_paths = find_all_paths(addresses=parent_dir[:2])
filtered_all_paths = filter_basedOn_labels(all_paths, label_df)
print(len(all_paths) - len(filtered_all_paths), 'studies are unlabeled.')
df = create_csv(filtered_all_paths, parent_dir, label_df, amy_folders)
df = filter_views(df, [ 'd', '2d', 'outflow', 'inflow', 'strain', 'contrast' ])
df['patient_id'] = df['path'].apply(lambda x:x.split('/')[-2])

# make data split
patient_df = group_patient(df)
stratify_features = encoding(patient_df)
train_df, val_df, test_df = create_split(patient_df, stratify_features, df)
make_csv(train_df, 'train')
make_csv(val_df, 'val')
make_csv(test_df, 'test')

# update directory
copy_files_by_split(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    amy_folders=amy_folders,
    source_root=parent_dir,     # Folder where all files currently live
    filepath_col='path',         
    output_root='data_split'    # Folder where you want to organize into train/val/test
)