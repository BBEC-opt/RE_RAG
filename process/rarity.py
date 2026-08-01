# Category rarity scoring based on TF-IDF.

import pandas as pd
import numpy as np
import math

class CategoryRarityCalculator:
    def __init__(self, alpha=0.8, normalize=True, norm_type="minmax", min_threshold=0.0):
        self.feature_rarity_scores = {}
        self.alpha = alpha
        self.normalize = normalize
        self.norm_type = norm_type
        self.min_threshold = min_threshold
        self.local_avgs = {}
        self.global_avgs = {}
    
    def calculate_rarity_scores(self, df, features):
        total_documents = len(df)
        all_rarity_values = []
        

        temp_scores = {}
        for feature in features:
            value_counts = df[feature].value_counts()
            document_freq = value_counts.to_dict()
            max_freq = value_counts.max()
            
            idf_scores = {}
            freq_rarity_list = []
            idf_list = []
            for value, freq in document_freq.items():
                idf = math.log(total_documents / (freq + 1)) + 1
                idf_scores[value] = idf
                idf_list.append(idf)
                
                freq_rarity = math.log(max_freq / (freq + 1)) + 1
                freq_rarity_list.append(freq_rarity)
            
            self.local_avgs[feature] = np.mean(idf_list)
            self.global_avgs[feature] = np.mean(freq_rarity_list)
            
            rarity_scores = {} 
            for value, freq in document_freq.items():
                freq_rarity = math.log(max_freq / (freq + 1)) + 1
                final_rarity = math.sqrt(idf_scores[value] * freq_rarity)
                rarity_scores[value] = final_rarity
                all_rarity_values.append(final_rarity)
            
            temp_scores[feature] = rarity_scores
        

        if self.normalize and len(all_rarity_values) > 0:
            if self.norm_type == "minmax":
                min_val, max_val = min(all_rarity_values), max(all_rarity_values)
                if max_val > min_val:
                    for feature in temp_scores:
                        temp_scores[feature] = {
                            k: max(self.min_threshold, (v - min_val) / (max_val - min_val))
                            for k, v in temp_scores[feature].items()
                        }

        
        self.feature_rarity_scores = temp_scores
        return self.feature_rarity_scores
    
    def get_rarity_score(self, feature, value):
        if feature in self.feature_rarity_scores and value in self.feature_rarity_scores[feature]:
            return self.feature_rarity_scores[feature][value]
        return self.min_threshold
    
    def add_rarity_features(self, df, features, suffix='_rarity'):
        df_copy = df.copy()
        for feature in features:
            if feature in self.feature_rarity_scores:
                new_col_name = f"{feature}{suffix}"
                df_copy[new_col_name] = df_copy[feature].map(
                    lambda x: self.get_rarity_score(feature, x)
                )
        return df_copy

if __name__ == "__main__":
    df = pd.read_pickle('../dataset/smpd/dataset_embed.pkl')
    categorical_features = ['user_id', 'pathalias', 'category','subcategory', 'concepts']
    calculator = CategoryRarityCalculator(min_threshold=0.1)
    calculator.calculate_rarity_scores(df, categorical_features)
    
    print("Average local and global rarity by feature:")
    for feature in categorical_features:
        local_avg = calculator.local_avgs.get(feature, 0)
        global_avg = calculator.global_avgs.get(feature, 0)
        print(f"{feature}: local={local_avg:.3f}, global={global_avg:.3f}")
    
    df_with_rarity = calculator.add_rarity_features(df, categorical_features)
    
    print(f"Data shape: {df_with_rarity.shape}")
    rarity_cols = [col for col in df_with_rarity.columns if '_rarity' in col]
    for col in rarity_cols:
        print(f"{col}: mean={df_with_rarity[col].mean():.3f}")

    output_path = '../dataset/smpd/dataset_q.pkl'
    df_with_rarity.to_pickle(output_path)
    print(df_with_rarity.columns)
    print(f"Output path: {output_path}")
