import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from angle_emb import AnglE

# 确保能读取损坏的图片文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= 配置路径 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PKL_PATH = '../dataset/smpd/dataset.pkl'
COVER_PATH = 'dataset/img/train'
OUT_PATH = '../dataset/smpd/dataset_embed.pkl'

# 模型本地路径
CLIP_MODEL_PATH = '../model/clip'
ANGLE_MODEL_PATH = '../model/bert'
# ============================================

def get_image_path(image_id, cover_path=COVER_PATH):
    """根据 image_id 生成图像的绝对路径"""
    image_rel_path = image_id.replace("_", "/")
    return os.path.join(cover_path, f"{image_rel_path}.jpg")


class TextEncoder:
    """文本编码器 (AnglE-BERT)"""
    def __init__(self, model_path, device):
        self.device = device
        self.angle = AnglE.from_pretrained(model_path, pooling_strategy='cls_avg')
        if "cuda" in str(self.device):
            self.angle = self.angle.cuda()

    def encode_text(self, texts, batch_size=32):
        text_list = texts.tolist() if isinstance(texts, pd.Series) else texts
        all_embeddings = []

        for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding text batches"):
            batch_texts = text_list[i:i+batch_size]
            # 确保输入是 list 结构
            batch_texts = [str(t) for t in batch_texts]
            batch_emb = self.angle.encode(batch_texts, to_numpy=True)
            all_embeddings.append(batch_emb)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.concatenate(all_embeddings, axis=0)


class CLIPImageEncoder:
    """图像编码器 (CLIP)"""
    def __init__(self, model_path, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

    def encode_images(self, img_paths, batch_size=32):
        features = []
        # 获取 CLIP 视觉特征维度，通常为 768 或 512
        feature_dim = self.model.config.projection_dim 

        for i in tqdm(range(0, len(img_paths), batch_size), desc="Encoding image batches"):
            batch_paths = img_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            # 加载图片并处理异常
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
                    continue
            
            # 如果整批图片都加载失败，直接用全零向量占位
            if not batch_images:
                features.append(torch.zeros(len(batch_paths), feature_dim))
                continue
                
            # 提取特征
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
            
            # 对齐批次大小，若有加载失败的图片，填充为0向量
            batch_results = torch.zeros(len(batch_paths), feature_dim).to(self.device)
            batch_results[valid_indices] = batch_features
            features.append(batch_results.cpu())
            
            # 释放显存
            del inputs, batch_features, batch_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(features).numpy()


if __name__ == '__main__':
    # 1. 读取数据
    print("Loading dataframe...")
    df = pd.read_pickle(PKL_PATH)

    # 2. 任务一：文本编码 (AnglE-BERT)
    print("\n--- Starting Text Encoding ---")
    text_encoder = TextEncoder(ANGLE_MODEL_PATH, DEVICE)
    text_features = text_encoder.encode_text(df['text'], batch_size=32)
    df['text_features'] = list(text_features)
    print(f"Text encoding completed. Shape: {text_features.shape}")

    # 3. 任务二：图像编码 (CLIP)
    print("\n--- Starting Image Encoding (CLIP) ---")
    img_paths = df['image_id'].apply(get_image_path).tolist()
    image_encoder = CLIPImageEncoder(CLIP_MODEL_PATH, DEVICE)
    image_features = image_encoder.encode_images(img_paths, batch_size=32)
    df['img_features'] = list(image_features)
    print(f"Image encoding completed. Shape: {image_features.shape}")

    # 4. 保存结果
    print(f"\nSaving results to {OUT_PATH}...")
    df.to_pickle(OUT_PATH)
    print("All tasks finished successfully!")