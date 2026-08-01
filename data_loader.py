import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class retrieval(Dataset):
    def __init__(self, pkl_file, top_k=5):
        super(retrieval, self).__init__()
        self.data = pd.read_pickle(pkl_file)
        self.top_k = top_k
        self._preload_base_features()
        self._preload_relation_features()
        self.video_id_to_idx = {vid: idx for idx, vid in enumerate(self.image_id)}

    def _preload_base_features(self):
        self.image_id = self.data['image_id'].values
        self.user_id = self.data['user_id'].values
        self.label = torch.tensor(self.data['label'].values,dtype=torch.float32)
        self.top_score =torch.tensor(self.data['top_score'],dtype=torch.float32)
        selected_columns = self.data[['category', 'subcategory', 'concepts', 'pathalias']]
        filled_data = selected_columns.fillna(0)
        self.bool_data = torch.as_tensor(
            filled_data.values.copy(), dtype=torch.float32
        )
        self.text_features = torch.tensor(
            np.stack(self.data['text_features']), 
            dtype=torch.float32
        )
        self.img_features = torch.tensor(
            np.stack(self.data['img_features']),
            dtype=torch.float32
        )

    def _preload_relation_features(self):
        self.top_video_ids = self.data['top_video_ids'].values

    def __len__(self):
        return len(self.data)

    def _get_features_by_ids(self,ori_id, video_ids, k,mode='pos'):
        features = {

            'user_id': [],
            'bool_data': [],
            'text_features':[],
            'img_features': [],

            'is_author':[],
            'label': []
        }
        current_user = self.user_id[ori_id]
        if mode == 'pos':
            score = self.top_score[ori_id][:self.top_k]
        for vid in video_ids[:k]:
            idx = self.video_id_to_idx[vid]
            features['user_id'].append(self.user_id[idx])
            features['bool_data'].append(self.bool_data[idx])
            features['text_features'].append(self.text_features[idx])
            features['img_features'].append(self.img_features[idx])


            features['label'].append(self.label[idx])

            features['is_author'].append(1 if self.user_id[idx] == current_user else 0)
        return {
            'text_features': torch.stack(features['text_features']),
            'img_features': torch.stack(features['img_features']),
            'bool_data': torch.stack(features['bool_data']),

            'user_id': torch.tensor(features['user_id'], dtype=torch.long),
            'score': score.clone().detach(),
            'label':torch.stack(features['label']),
            'is_author': torch.tensor(features['is_author'], dtype=torch.long),
        }

    def __getitem__(self, idx):
        base_data = {
            'image_id': self.image_id[idx],
            'user_id': self.user_id[idx],
            'bool_data': self.bool_data[idx],
            'text_features': self.text_features[idx],
            'img_features': self.img_features[idx],

            'label': self.label[idx],
        }
        top_data = self._get_features_by_ids(idx,self.top_video_ids[idx], self.top_k)
        top_data = {f'top_{k}': v for k, v in top_data.items()}
        return {
            'base_data': base_data,
            'top_data': top_data,
            'pp_label': self.label[idx]
        }


def load_data(args):
    random_seed = 2025
    random_generator = random.Random(random_seed)
    full_dataset = retrieval(
        args.data_files,
        top_k=args.top_k,
    )
    indices = list(range(len(full_dataset)))
    random_generator.shuffle(indices)
    test_size = int(len(full_dataset) *(1-args.train_ratio)*0.5)    
    val_size = int(len(full_dataset)*(1-args.train_ratio)*0.5)        
    train_size = len(full_dataset)-val_size-test_size  
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size+val_size]
    train_indices = indices[test_size+val_size:]
    

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)
    loader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': args.num_workers > 0,
        'drop_last': True
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    
    print(
        f'Dataset split complete | train: {len(train_set)} | '
        f'validation: {len(val_set)} | test: {len(test_set)}'
    )
    return train_loader, val_loader, test_loader
