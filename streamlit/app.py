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

def get_single_image_embedding(my_image,model,processor,device):
    image = processor(text = None,images = my_image, return_tensors="pt")["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np

def load_image_upload():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_blip_prompt(image):
    # prompt = "Question: which city is this? Answer:"
    # prompt = input ("Enter the Question: ")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b",torch_dtype=torch.float16)
    model= Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = st.text_input('Enter the Question: ')
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    st.markdown(generated_text)
    # print(generated_text)

def load_clip_similar(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ID = "openai/clip-vit-base-patch32"
    model, processor, tokenizer = get_model_info(model_ID, device)
    image_data_df = pd.read_pickle('/home/ubuntu/git_repo/Clip_Image_Search_Model/data/df')
    query_caption = image
    top_images = get_top_N_images(query_caption, image_data_df,model,processor,device)
    plot_images_by_side(top_images)


def plot_images(images):

  for image in images:
    plt.imshow(image)
    plt.show()

def plot_images_by_side(top_images):

  index_values = list(top_images.index.values)
  list_images = [top_images.iloc[idx].image for idx in index_values] 
  list_captions = [top_images.iloc[idx].caption for idx in index_values] 
  similarity_score = [top_images.iloc[idx].cos_sim for idx in index_values] 

  n_row = n_col = 2

  _, axs = plt.subplots(n_row, n_col, figsize=(15, 15))
  [axi.set_axis_off() for axi in axs.ravel()]
  axs = axs.flatten()
  for img, ax, caption, sim_score in zip(list_images, axs, list_captions, similarity_score):
      ax.imshow(img)
      sim_score = 100*float("{:.2f}".format(sim_score))
      # ax.title.set_text(f"Caption: {caption}\nSimilarity: {sim_score}%")
  plt.show()

def get_top_N_images(query, data, model,processor,device,top_K=4):
    query_vect = get_single_image_embedding(query,model,processor,device)
    revevant_cols = ["caption", "image", "cos_sim"]
    data["cos_sim"] = data["img_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))
    data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])
    most_similar_articles = data.sort_values(by='cos_sim', ascending=False)[1:top_K+1]
    return most_similar_articles[revevant_cols].reset_index()

def main():
    st.title('Image Search Question Answering using BLIP_2')

    img = load_image_upload()
    if img != None :
        load_blip_prompt(img)
#         load_clip_similar(img)

if __name__ == '__main__':
    main()
