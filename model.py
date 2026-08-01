# RGP model and its required components.

import math

import torch
import torch.nn.functional as F
from torch import nn


def add_positional_encoding(x):
    bs, n, dim = x.size()
    pe = torch.zeros(n, dim, device=x.device)
    position = torch.arange(0, n, dtype=torch.float, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=x.device).float() * (-math.log(10000.0) / dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0).expand(bs, -1, -1)  # (bs, n, dim)
    return x + pe

class embed_relation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.relation_embedding1 = nn.Embedding(3, 128)
        self.relation_embedding2 = nn.Embedding(3, 128)
        self.relation_embedding3 = nn.Embedding(3, 128)
        self.relation_embedding4 = nn.Embedding(3, 128)
        self.relation_embedding5 = nn.Embedding(3, 128)

        self.relation_projection = nn.Sequential(
            nn.Linear(128*5, 768),
            nn.ReLU(),
            nn.Linear(768, d_model)
        )
    def forward(self, author_ids, anchor_author_id, category, anchor_cat, subcategory, anchor_sub, concepts, anchor_con, is_pro, anchor_is_pro):
        bs, n = author_ids.shape
        
        author_same = (author_ids.unsqueeze(1) == author_ids.unsqueeze(2))
        is_anchor = (author_ids == anchor_author_id.unsqueeze(1))
        
        author_mask = torch.zeros_like(author_same, dtype=torch.long)
        author_mask[author_same] = 1
        author_mask[is_anchor.unsqueeze(1) | is_anchor.unsqueeze(2)] = 2

        pro_same = (is_pro.unsqueeze(1) == is_pro.unsqueeze(2))
        anchor_pro_mask = (is_pro == anchor_is_pro.unsqueeze(1))
        
        pro_mask = torch.zeros_like(pro_same, dtype=torch.long)
        pro_mask[pro_same] = 1
        pro_mask[anchor_pro_mask.unsqueeze(1) | anchor_pro_mask.unsqueeze(2)] = 2

        cat_same = (category.unsqueeze(1) == category.unsqueeze(2))
        anchor_cat_mask = (category == anchor_cat.unsqueeze(1))
        
        cat_mask = torch.zeros_like(cat_same, dtype=torch.long)
        cat_mask[cat_same] = 1
        cat_mask[anchor_cat_mask.unsqueeze(1) | anchor_cat_mask.unsqueeze(2)] = 2
        
        subcat_same = (subcategory.unsqueeze(1) == subcategory.unsqueeze(2))
        anchor_subcat_mask = (subcategory == anchor_sub.unsqueeze(1))
        
        subcat_mask = torch.zeros_like(subcat_same, dtype=torch.long)
        subcat_mask[subcat_same] = 1
        subcat_mask[anchor_subcat_mask.unsqueeze(1) | anchor_subcat_mask.unsqueeze(2)] = 2
        
        concept_same = (concepts.unsqueeze(1) == concepts.unsqueeze(2))
        anchor_concept_mask = (concepts == anchor_con.unsqueeze(1))
        
        concept_mask = torch.zeros_like(concept_same, dtype=torch.long)
        concept_mask[concept_same] = 1
        concept_mask[anchor_concept_mask.unsqueeze(1) | anchor_concept_mask.unsqueeze(2)] = 2
        
        author_emb = self.relation_embedding1(author_mask.long())      # [bs, n, n, 64]
        pro_emb = self.relation_embedding2(pro_mask.long())            # [bs, n, n, 64]
        cat_emb = self.relation_embedding3(cat_mask.long())            # [bs, n, n, 64]
        subcat_emb = self.relation_embedding4(subcat_mask.long())      # [bs, n, n, 64]
        concept_emb = self.relation_embedding5(concept_mask.long())    # [bs, n, n, 64]

        combined_emb = torch.cat([author_emb, pro_emb, cat_emb, subcat_emb, concept_emb], dim=-1)  # [bs, n, n, 64*5]
        projected_emb = self.relation_projection(combined_emb)  # [bs, n, n, d_model]
        return projected_emb, is_anchor, anchor_pro_mask, anchor_cat_mask, anchor_subcat_mask, anchor_concept_mask


class AuthorAwareCrossAttention3(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
        self.norm = nn.LayerNorm(d_model)

        self.embed_rel = embed_relation(d_model)

        self.relation_embedding1 = nn.Embedding(2, 128)
        self.relation_embedding2 = nn.Embedding(2, 128)
        self.relation_embedding3 = nn.Embedding(2, 128)
        self.relation_embedding4 = nn.Embedding(2, 128)
        self.relation_embedding5 = nn.Embedding(2, 128)

        self.relation_projection = nn.Sequential(
            nn.Linear(128*5, 768),
            nn.ReLU(),
            nn.Linear(768, d_model)
        )
    def forward(self, anchor_features, features, is_anchor, anchor_pro_mask, anchor_cat_mask, anchor_subcat_mask, anchor_concept_mask):
        bs, n, _ = features.shape
        device = features.device

        Q = self.W_q(anchor_features).unsqueeze(1)  # [bs, 1, d_model]
        K = self.W_k(features)                      # [bs, n, d_model]
        V = self.W_v(features)                      # [bs, n, d_model]

        emb1 = self.relation_embedding1(is_anchor.long())          # [bs, n, 64]
        emb2 = self.relation_embedding2(anchor_pro_mask.long())    # [bs, n, 64]
        emb3 = self.relation_embedding3(anchor_cat_mask.long())    # [bs, n, 64]
        emb4 = self.relation_embedding4(anchor_subcat_mask.long()) # [bs, n, 64]
        emb5 = self.relation_embedding5(anchor_concept_mask.long())# [bs, n, 64]
        combined_emb = torch.cat([emb1, emb2, emb3, emb4, emb5], dim=-1)  # [bs, n, 64*5]
        rel_emb = self.relation_projection(combined_emb)           # [bs, n, d_model]

        Q = Q.view(bs, 1, self.n_head, self.head_dim).transpose(1, 2)  # [bs, head, 1, d]
        K = K.view(bs, n, self.n_head, self.head_dim).transpose(1, 2)  # [bs, head, n, d]
        V = V.view(bs, n, self.n_head, self.head_dim).transpose(1, 2)  # [bs, head, n, d]
        rel = rel_emb.view(bs, n, self.n_head, self.head_dim).transpose(1, 2) 

        K_fused = K * rel  # [bs, head, n, d]

        attn_scores = torch.matmul(Q, K_fused.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [bs, head, 1, n]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [bs, head, 1, n]
        attn_output = torch.matmul(attn_weights, V)    # [bs, head, 1, d]

        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, 1, self.d_model)  # [bs, 1, d_model]
        attn_output = self.W_o(attn_output).squeeze(1)  # [bs, d_model]
        attn_output = self.norm(attn_output + anchor_features)
        return attn_output, attn_weights


class AuthorAwareAttention3(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
        self.embed_rel = embed_relation(d_model)

        self.W_rel = nn.Linear(d_model, n_head)

        self.cross_attn = AuthorAwareCrossAttention3(d_model, n_head)

    def forward(self, anchor_features, features, author_ids, anchor_author_id,category,anchor_cat, subcategory,anchor_sub, concepts,anchor_con, is_pro,anchor_is_pro):
        bs, n, _ = features.shape
        
        features = add_positional_encoding(features)
        Q = self.W_q(features)
        K = self.W_k(features)
        V = self.W_v(features)
        rel_emb, is_anchor, anchor_pro_mask, anchor_cat_mask, anchor_subcat_mask, anchor_concept_mask = \
            self.embed_rel(author_ids, anchor_author_id, category, anchor_cat, subcategory, anchor_sub, concepts, anchor_con, is_pro, anchor_is_pro)
        
        Q = Q.view(bs, n, self.n_head, self.head_dim).transpose(1, 2)  # [bs, head, n, head_dim]
        K = K.view(bs, n, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(bs, n, self.n_head, self.head_dim).transpose(1, 2)
        
        # K: [bs, head, n, d]
        K_t = K.transpose(1, 2)

        rel_exp = rel_emb.view(bs, n, n, self.n_head, self.head_dim)  # [bs, n, n, head, d]
        K_exp = K_t.unsqueeze(1)        # [bs, 1, n, head, d]

        fused = rel_exp * K_exp  # [bs, n, n, head, d]

        K_struct = fused.permute(0, 3, 1, 2, 4)

        # Q: [bs, head, n, d]
        attn_scores = (Q.unsqueeze(3) * K_struct).sum(-1) / (self.head_dim ** 0.5)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [bs, head, n, head_dim]
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, n, self.d_model)
        attn_output = self.norm(self.W_o(attn_output)+features)

        cross_features, cross_attn_weights = self.cross_attn(anchor_features, attn_output, is_anchor, anchor_pro_mask, anchor_cat_mask, anchor_subcat_mask, anchor_concept_mask)

        return cross_features, cross_attn_weights


class CoAttention(nn.Module):
    def __init__(self, input_dim, out_dim=None, num_heads=8, dropout=0.1):
        super(CoAttention, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim if out_dim is not None else input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout = dropout
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.img2text_q = nn.Linear(input_dim, input_dim, bias=False)
        self.img2text_k = nn.Linear(input_dim, input_dim, bias=False)
        self.img2text_v = nn.Linear(input_dim, input_dim, bias=False)
        self.img2text_out = nn.Linear(input_dim, input_dim)
        
        self.text2img_q = nn.Linear(input_dim, input_dim, bias=False)
        self.text2img_k = nn.Linear(input_dim, input_dim, bias=False)
        self.text2img_v = nn.Linear(input_dim, input_dim, bias=False)
        self.text2img_out = nn.Linear(input_dim, input_dim)
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),
            nn.ReLU(),
            nn.Linear(out_dim, self.out_dim)
        )

    def cross_attention(self, query, key, value, q_proj, k_proj, v_proj, out_proj):
        batch_size = query.size(0)
        
        Q = q_proj(query)  # [B, input_dim]
        K = k_proj(key)    # [B, input_dim]
        V = v_proj(value)  # [B, input_dim]
        
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # [B, num_heads, head_dim]
        K = K.view(batch_size, self.num_heads, self.head_dim)  # [B, num_heads, head_dim]
        V = V.view(batch_size, self.num_heads, self.head_dim)  # [B, num_heads, head_dim]
        
        scores = torch.sum(Q * K, dim=-1) / math.sqrt(self.head_dim)  # [B, num_heads]
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads]
        attn_weights = self.attn_dropout(attn_weights)

        attn_weights = attn_weights.unsqueeze(-1)  # [B, num_heads, 1]
        context = attn_weights * V  # [B, num_heads, head_dim]
        
        context = context.view(batch_size, self.input_dim)  # [B, input_dim]
    
        output = out_proj(context)
        output = self.out_dropout(output)

        return output

    def forward(self, img_features, text_features):
        """
        img_features: [batch_size, input_dim]
        text_features: [batch_size, input_dim]
        """

        img_attended = self.cross_attention(
            query=img_features, 
            key=text_features, 
            value=text_features,
            q_proj=self.img2text_q,
            k_proj=self.img2text_k,
            v_proj=self.img2text_v,
            out_proj=self.img2text_out
        )
        img_enhanced = self.norm1(img_features + img_attended)
        
        text_attended = self.cross_attention(
            query=text_features, 
            key=img_features, 
            value=img_features,
            q_proj=self.text2img_q,
            k_proj=self.text2img_k,
            v_proj=self.text2img_v,
            out_proj=self.text2img_out
        )
        text_enhanced = self.norm2(text_features + text_attended)
        
        fused = torch.cat([img_enhanced, text_enhanced], dim=1)  # [B, 2*dim]

        output = self.fusion_layer(fused)  # [B, out_dim]
        
        return output


class RGP(nn.Module):
    def __init__(self, batch_size,top_k):
        super().__init__()
        self.use_k2=top_k
        self.batch_size = batch_size
        drop_num = 0.2
        self.drop = nn.Dropout(drop_num)
        self.img_MLP = nn.Sequential(
            nn.Linear(768, 512),

            nn.ReLU(),
            nn.Linear(512, 384),
        )
        # textual MLP
        self.text_MLP = nn.Sequential(
            nn.Linear(768, 512),

            nn.ReLU(),
            nn.Linear(512, 384),
        )

        self.coattention = CoAttention(input_dim=384,out_dim=768)
        self.prediction = nn.Sequential(
            nn.Linear(768+768, 1024),

            nn.ReLU(),
            nn.Linear(1024, 512),

            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.context_attn = AuthorAwareAttention3(d_model=768, n_head=4)

        self.pos_mlp = nn.Sequential(
            nn.Linear(768+3+1, 1024),

            nn.ReLU(),
            nn.Linear(1024, 768),
        )
    def get_features(self, text_features, img_features):
        img_features = self.img_MLP(img_features)
        text_features = self.text_MLP(text_features)

        fused_features = self.coattention(img_features,text_features)
        return fused_features

    def batch_feature_extraction(self, text_features, img_features,):
        batch_size, k = text_features.shape[:2]
        flat = lambda x: x.view(-1, *x.shape[2:])
        fused_features =self.get_features(
            flat(text_features), flat(img_features)
        )
        fused_features=fused_features.view(batch_size,k,-1)

        return fused_features

    def forward(self, text_features,img_features,bool_data,user_id,
                    top_text_features,top_img_features,top_bool_data,top_user_id,
                    top_score,top_label
                    ):
        anchor_features = self.get_features(text_features, img_features)
        positives_features = self.batch_feature_extraction(top_text_features, top_img_features)

        combined_sim=top_score
        top_label = top_label.unsqueeze(-1)

        pos_features = torch.cat([
            positives_features,
            combined_sim,
            top_label,
        ], dim=-1)
        pos_features = self.pos_mlp(pos_features)  # [bs, k, 768]

        category = top_bool_data[:,:,0]
        anchor_cat = bool_data[:,0]
        subcategory = top_bool_data[:,:,1]
        anchor_sub = bool_data[:,1]
        concepts = top_bool_data[:,:,2]
        anchor_con = bool_data[:,2]
        is_pro = top_bool_data[:,:,3]
        anchor_is_pro = bool_data[:,3]


        features, _ = self.context_attn(anchor_features,pos_features,top_user_id,user_id,\
                                     category,anchor_cat, subcategory,anchor_sub, concepts,anchor_con, is_pro,anchor_is_pro)

        feat = torch.cat([
            anchor_features,
            features,
        ], dim=1)

        out = self.prediction(feat)

        return out
