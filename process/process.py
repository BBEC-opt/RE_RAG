
import pandas as pd
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re


def encode_tags(word_list):
    word_dict = {}
    encoded_list = []

    for sublist in word_list:

        if sublist == []:
            encoded_list.append([0])
            continue

        words = []

        if sublist not in word_dict.keys():
            word_dict[sublist] = len(word_dict) + 1

        #words.append(word_dict[sublist])
        encoded_list.append(word_dict[sublist])

    return encoded_list
def generate_structured_text(title,Category,Subcategory,Concept,Pathalias,Uid,tags):
    # 字段清洗
    # title = re.sub(r'[《》「」]', '', title).strip()
    # tags = ', '.join(list(set(tags.split(',')))) if tags else "[null]"

    return f"title: {title} [SEP] Category: {Category} [SEP] Subcategory: {Subcategory} [SEP] Concept: {Concept} [SEP] Pathalias: {Pathalias} [SEP] tags: {tags} [SEP] author: {Uid}"

def text_hand(title,Category,Subcategory,Concept,Pathalias,Uid):
    text = title + " " + Category + " " + Subcategory + " " + Concept + " " + Pathalias + " " + Uid

    text = text.replace("\n", " ").replace("\r", "").replace("\t", "")

    return text

def process_meta_data(train_text, train_additional_information, train_category, train_temporalspatial_information, train_user_data, label_data, path):

    # all_tags = [tags.split() for tags in train_text['Alltags']]
    all_tags = []
    tag_len = []

    for tags in train_text['Alltags']:
        all_tags.append(tags.split())
        tag_len.append(len(tags.split()))

    # text = train_text['Title']
    #text = generate_structured_text(train_text['Title'], train_category['Category'], train_category['Subcategory'], train_category['Concept'], train_additional_information['Pathalias'], train_additional_information['Uid'],train_text['Alltags'])
    texts = [
        generate_structured_text(t, c, sc, con,p,u,ta)
        for t, c, sc, con,p,u,ta in zip(train_text['Title'], train_category['Category'], train_category['Subcategory'], train_category['Concept'], train_additional_information['Pathalias'], train_additional_information['Uid'],train_text['Alltags'])
    ]
    pattern = r'<a[^>]*>(.*?)</a>'

    clean_text = [re.sub(pattern, '', item) for item in texts]
    pathalias = encode_tags(train_additional_information['Pathalias'])

    image_id = [f"{x}_{y}" for x, y in zip(train_additional_information['Uid'], train_additional_information['Pid'])]

    user_id = encode_tags(train_additional_information['Uid'])

    category = encode_tags(train_category['Category'])

    subcategory = encode_tags(train_category['Subcategory'])

    concepts = encode_tags(train_category['Concept'])

    title_length = train_text['Title'].str.len()
    tag_num = tag_len
    user_description_len = train_user_data['user_description'].str.len().fillna(0)
    photo_count = train_user_data['photo_count'].fillna(0)
    scaler_meta = MinMaxScaler()
    meta_features = np.array([
        title_length,
        tag_num,
        user_description_len,
        photo_count,
    ]).T  # 转置为 (n_samples, n_features)
    normalized_meta = scaler_meta.fit_transform(meta_features)

    # time_zone = encode_tags(train_user_data['time_zone_offset'])
    time_zone_offset = encode_tags(train_user_data['timezone_offset'])
    postdate = train_temporalspatial_information['Postdate']

    normalized_title_length = normalized_meta[:, 0]
    normalized_tag_num = normalized_meta[:, 1]
    normalized_user_description_len = normalized_meta[:, 2]
    normalized_photo_count = normalized_meta[:, 3]
    time_zone_id = train_user_data['timezone_id']
    zone = encode_tags(time_zone_id)
    is_pro = np.nan_to_num(train_user_data['ispro'].values, nan=0).tolist()
    label = label_data[0].tolist()

    dataset = {
        'image_id': image_id,
        'text': clean_text,
        'tags': all_tags,
        'label': label,
        'user_id': user_id,
        'pathalias': pathalias,
        'category': category,
        'subcategory': subcategory,
        'concepts': concepts,
        'time_zone_id': zone,
        'time_zone_offset': time_zone_offset,
        'is_pro':is_pro,
    }

    df = pd.DataFrame(dataset)

    df.to_pickle(path)

    return df

if __name__ == "__main__":
    path = '../dataset/smpd/dataset.pkl'
    origin_data_path = 'dataset'

    train_additional_information = pd.read_json(os.path.join(origin_data_path, 'train_additional_information.json'))
    train_category = pd.read_json(os.path.join(origin_data_path, 'train_category.json'))
    train_temporalspatial_information = pd.read_json(
        os.path.join(origin_data_path, 'train_temporalspatial_information.json'))
    train_user_data = pd.read_json(os.path.join(origin_data_path, 'train_user_data.json'))
    label_data = pd.read_csv(os.path.join(origin_data_path, 'train_label.txt'), header=None)
    train_text = pd.read_json(os.path.join(origin_data_path, 'train_text.json'))

    process_meta_data(train_text, train_additional_information, train_category, train_temporalspatial_information, train_user_data, label_data, path)
    print("Meta data processed!")





