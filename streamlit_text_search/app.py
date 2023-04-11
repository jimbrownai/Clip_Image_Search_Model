import os
import torch
import skimage
import requests
import numpy as np
import pandas as pd
from PIL import Image
import io
# from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st
import json

from datasets import load_dataset
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity  

from transformers import AutoTokenizer,AutoProcessor
from transformers import Blip2ForConditionalGeneration


def get_model_info(model_ID, device):

  model = CLIPModel.from_pretrained(model_ID).to(device)
  processor = CLIPProcessor.from_pretrained(model_ID)
  tokenizer = CLIPTokenizer.from_pretrained(model_ID)
  return model, processor, tokenizer

def set_global_variables():
    global device
    global model
    global processor
    global tokenizer
    global cocopath
    cocopath = 'D:\git\CLIP_prefix_caption\data\coco'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ID = "openai/clip-vit-base-patch32"
    model,processor, tokenizer = get_model_info(model_ID, device)


def get_single_image_embedding(my_image):
    image = processor(text = None,images = my_image, return_tensors="pt")["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np

def get_all_images_embedding(df, img_column):
    df["img_embeddings"] = df[str(img_column)].apply(get_single_image_embedding)
    return df

def get_single_text_embedding(text):
    inputs = tokenizer(text, return_tensors = "pt").to(device)
    text_embeddings = model.get_text_features(**inputs)
    # convert the embeddings to numpy array 
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np

def get_all_text_embeddings(df, text_col):
    df["text_embeddings"] = df[str(text_col)].apply(get_single_text_embedding)
    return df 

def load_and_process_data(limit):
    f = open(r'D:\git\CLIP_prefix_caption\data\coco\annotations\train_caption.json')
    coco_data = json.load(f)
    data_dict = {'image_id':[],'image':[],'caption':[]}
    # print(limit)
    for i in range(limit):
        d = coco_data[i]
        img_id = d["image_id"]
        filename = f"D:/git/CLIP_prefix_caption/data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"D:/git/CLIP_prefix_caption/data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = Image.open((filename)).convert("RGB")
        data_dict['image_id'].append(img_id)
        data_dict['image'].append(image)
        data_dict['caption'].append(d['caption'])
        img_df = pd.DataFrame(data_dict)
    print(len(img_df['image_id']))
    return img_df

def create_text_image_embedding(img_df):
    img_df = get_all_text_embeddings(img_df, "caption")
    img_df = get_all_images_embedding(img_df, "image")

    return img_df

def get_top_N_images(query, data,top_K=4):
    # query_vect = get_single_image_embedding(query)
    query_vect = get_single_text_embedding(query)
    revevant_cols = ["caption", "image", "cos_sim"]
    data["cos_sim"] = data["img_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))
    data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])
    most_similar_articles = data.sort_values(by='cos_sim', ascending=False)[1:top_K+1]
    return most_similar_articles[revevant_cols].reset_index()

def remove_duplicates(imgs):
    cos_sim = set()
    res_data = {'index':[],'caption':[],'image':[],'cos_sim':[]}
    for index, row in imgs.iterrows():
        if row['cos_sim'] in cos_sim :
            continue
        else:
            cos_sim.add(row['cos_sim'])
            res_data['index'].append(row['index'])
            res_data['caption'].append(row['caption'])
            res_data['image'].append(row['image'])
            res_data['cos_sim'].append(row['cos_sim'])
    return res_data

def main():
    set_global_variables()
    # Setting db limit
    df = load_and_process_data(500)
    print('Data Loaded')
    df = create_text_image_embedding(df)
    print('Created Text and Image Embedding')
    st.title('Image Search Using Clip')
    query = st.text_input('Type to Search', 'Car')
    st.write('The current input query is:', query)
    imgs = get_top_N_images(query, df,top_K=15)
    print('Retrived Images')
    rs = remove_duplicates(imgs)
    st.image(rs['image'])
    # df.shape[0]
    # st.dataframe(imgs)
    
    

if __name__ == '__main__':
    main()
