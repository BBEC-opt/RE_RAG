# Sparse multimodal retrieval.

from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm

class VideoRetriever:
    def __init__(self, data_path, device='cuda', feature_weights=None):

        self.df = pd.read_pickle(data_path)

        

        self.device = device


        self.feature_weights = self._validate_weights(feature_weights or {'text':0.0, 'img':0.0})
        

        self.image_id = self.df['image_id'].values
        self.label = self.df['label'].values


        self.user_id_encode = torch.tensor(
            self.df['user_id'].values,
            device=self.device
        )
        self.pathalias = torch.tensor(
            self.df['pathalias'].values,
            device=self.device
        )
        self.category = torch.tensor(
            self.df['category'].values,
            device=self.device
        )
        self.subcategory = torch.tensor(
            self.df['subcategory'].values,
            device=self.device
        )
        self.concepts = torch.tensor(
            self.df['concepts'].values,
            device=self.device
        )

        self.user_id_rarity = torch.tensor(
            self.df['user_id_rarity'].values, 
            device=self.device
        )
        self.category_rarity = torch.tensor(
            self.df['category_rarity'].values, 
            device=self.device
        )
        self.subcategory_rarity = torch.tensor(
            self.df['subcategory_rarity'].values, 
            device=self.device
        )
        self.pathalias_rarity = torch.tensor(
            self.df['pathalias_rarity'].values, 
            device=self.device
        )
        self.concepts_rarity = torch.tensor(
            self.df['concepts_rarity'].values, 
            device=self.device
        )
        
        self._load_core_features()

    def _encode_authors(self):
        if 'user_id' not in self.df.columns:
            raise KeyError("Missing required 'author' column")
            

        from sklearn.preprocessing import LabelEncoder
        self.author_encoder = LabelEncoder()
        self.df['user_id_x'] = self.author_encoder.fit_transform(self.df['user_id'])
    def _process_time_data(self):

        self.df['publish_time'] = pd.to_datetime(
            self.df['publish_time'].apply(lambda x: datetime(*x)),
            format='%Y-%m-%d'
        )
        reference_date = pd.Timestamp('1970-01-01')
        self.df['publish_diff_days'] = (self.df['publish_time'] - reference_date).dt.days

    def _validate_weights(self, weights):
        valid_modes = ['text', 'img']
        for k in valid_modes:
            if k not in weights:
                raise ValueError(f"Missing weight for {k}")
            if weights[k] < 0:
                raise ValueError(f"Negative weight for {k}")
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _load_core_features(self):
        self.features = {}
        core_modes = {
            'text': ('text_features', 768),
            'img': ('img_features', 768),
        }
        for mode,  (col, expected_dim) in core_modes.items():
            valid_features = []
            invalid_indices = []
            feat = torch.stack([torch.tensor(f) for f in self.df[col]])
            if invalid_indices:
                print(
                    f"Mode {mode}: found {len(invalid_indices)} invalid features; "
                    f"first 10 indices: {invalid_indices[:10]}"
                )
            self.features[mode] = torch.nn.functional.normalize(feat.squeeze(), p=2, dim=1).to(self.device)

    def batch_retrieve(self,top_k=10, batch_size=512,output_path=None):
        results = []
        total = len(self.df)
        
        with torch.no_grad():
            for i in tqdm(range(0, total, batch_size), desc="Retrieving videos", unit="batch"):
                batch_end = min(i + batch_size, total)
                current_features = {mode: self.features[mode][i:batch_end] for mode in self.features}
                

                total_sim, mode_sims = self._compute_similarity(current_features)
                

                rows = torch.arange(batch_end - i, device=self.device)
                cols = i + torch.arange(0, batch_end - i, device=self.device)
                mask = torch.ones_like(total_sim, dtype=torch.bool)
                mask[rows, cols] = False


                current_categories = self.category[i:batch_end]

                category_mask = current_categories.unsqueeze(1) == self.category.unsqueeze(0)
                current_category_rarity = self.category_rarity[i:batch_end]
                category_coeff = torch.where(
                    category_mask,
                    1.0+current_category_rarity.unsqueeze(1),
                    torch.tensor(1.0, device=self.device)
                )
                total_sim = total_sim * category_coeff
                

                current_subcategories = self.subcategory[i:batch_end]

                subcategory_mask = current_subcategories.unsqueeze(1) == self.subcategory.unsqueeze(0)
                current_subcategory_rarity = self.subcategory_rarity[i:batch_end]
                subcategory_coeff = torch.where(
                    subcategory_mask,
                    1.0+current_subcategory_rarity.unsqueeze(1),
                    torch.tensor(1.0, device=self.device)
                )
                total_sim = total_sim * subcategory_coeff
                
                current_pathalias = self.pathalias[i:batch_end]

                pathalias_mask = current_pathalias.unsqueeze(1) == self.pathalias.unsqueeze(0)
                current_pathalias_rarity = self.pathalias_rarity[i:batch_end]
                pathalias_coeff = torch.where(
                    pathalias_mask,
                    1.0+current_pathalias_rarity.unsqueeze(1),
                    torch.tensor(1.0, device=self.device)
                )
                total_sim = total_sim * pathalias_coeff

                current_concepts = self.concepts[i:batch_end]

                concepts_mask = current_concepts.unsqueeze(1) == self.concepts.unsqueeze(0)
                current_concepts_rarity = self.concepts_rarity[i:batch_end]
                concepts_coeff = torch.where(
                    concepts_mask,
                    1.0+current_concepts_rarity.unsqueeze(1),
                    torch.tensor(1.0, device=self.device)
                )
                total_sim = total_sim * concepts_coeff
                
                current_authors = self.user_id_encode[i:batch_end]
                author_mask = current_authors.unsqueeze(1) == self.user_id_encode.unsqueeze(0)
                current_author_rarity = self.user_id_rarity[i:batch_end]
                author_coeff = torch.where(
                    author_mask,
                    1.0+current_author_rarity.unsqueeze(1),
                    torch.tensor(1.0, device=self.device)
                )
                total_sim = total_sim * author_coeff


                masked_sim_top = total_sim.masked_fill(~mask, -torch.inf)
                top_scores, top_indices1 = torch.topk(masked_sim_top, k=top_k, dim=1)


                text_scores = torch.gather(mode_sims.get('text', torch.zeros_like(total_sim)), 1, top_indices1)
                img_scores = torch.gather(mode_sims.get('img', torch.zeros_like(total_sim)), 1, top_indices1)


                text_cpu = text_scores.cpu().numpy()
                img_cpu = img_scores.cpu().numpy()

                indices_cpu = top_indices1.cpu().numpy()
                scores_cpu = top_scores.cpu().numpy()
                

                if self.feature_weights['text'] > 0:
                    text_cpu = text_cpu / self.feature_weights['text']
                if self.feature_weights['img'] > 0:
                    img_cpu = img_cpu / self.feature_weights['img']

                for j in range(batch_end - i):
                    original_idx = i + j
                    result = self._format_result(
                        original_idx=original_idx, 
                        top_indices=indices_cpu[j], 
                        top_scores=scores_cpu[j],
                        text_top=text_cpu[j],
                        img_top=img_cpu[j],
                    )
                    results.append(result)
            self.save_results(pd.DataFrame(results),output_path=output_path)
            print("Retrieval completed and results saved.")

    def _compute_similarity(self, batch_features):
        total_sim = 0
        mode_sims = {}

        for mode in self.feature_weights:
            if self.feature_weights[mode] <= 0:
                continue
            feat = self.features[mode]
            batch_feat = batch_features[mode]
            raw_sim = torch.mm(batch_feat, feat.T)
                
            weighted_sim = self.feature_weights[mode] * raw_sim
            mode_sims[mode] = weighted_sim
            total_sim += weighted_sim

        if total_sim is None:
            raise ValueError("Failed to compute total similarity; check the input data")

        return total_sim, mode_sims

    def _format_result(self, original_idx, top_indices, top_scores, text_top,img_top):
        result = {
            "image_id": self.image_id[original_idx],
            "top_score": [[float(s), float(txt), float(img)] 
                        for s, txt, img in zip(top_scores, text_top, img_top,)],
            "top_video_ids": [self.image_id[i] for i in top_indices],
        }
        return result

    def save_results(self, sim_df,output_path):

        original_df = self.df.copy(deep=True)
        

        original_df.set_index('image_id',inplace=True, verify_integrity=False)
        sim_df.set_index('image_id',inplace=True, verify_integrity=False)


        missing_ids = original_df.index.difference(sim_df.index)
        if not missing_ids.empty:
            print(f"Warning: {len(missing_ids)} video IDs are missing from sim_df")

        sim_df = sim_df.reindex(original_df.index)
    
        columns_to_merge = ['top_score', 'top_video_ids']

        original_df = original_df.drop(columns=[col for col in columns_to_merge if col in original_df.columns], errors='ignore')

        merged_df = original_df.merge(
            sim_df[['top_score', "top_video_ids",]],
            how='left',
            left_index=True,
            right_index=True,
        ).reset_index()


        nan_columns = merged_df.columns[merged_df.isna().any()].tolist()
        if nan_columns:
            print(f"Warning: the following columns contain NaN values: {nan_columns}")

            merged_df['top_score'] = merged_df['top_score'].apply(
                lambda x: x if isinstance(x, list) else []
            )
        
        merged_df.to_pickle(output_path)
        print(f"Saved {len(merged_df)} records; columns containing NaN: {nan_columns}")


if __name__ == "__main__":
    data_file = '../dataset/smpd/dataset_q.pkl'

    retriever = VideoRetriever(
        data_path=data_file,
        feature_weights={'text': 0.8, 'img': 0.2},
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    result_df = retriever.batch_retrieve(top_k=70, batch_size=512, output_path=data_file)
