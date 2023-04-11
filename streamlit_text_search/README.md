# Streamlit App to Retrive Images for the given Input Text Query
Dataset - Coco2014, Stored in local path -> to dowload : https://cocodataset.org/#download
Model - CLIP "openai/clip-vit-base-patch32"

## Image and Text Embedding
Created text and Image embedding using the model processor (clip processor) for the given limit of images and their captions
    Functions Implemented - > load_and_process_data(500), create_text_image_embedding(df) 

## Image retrival and sorting
Implemented the cosine similarity function from project 1 and created functions to select distnict images from the retrived query 
    Functions ->  get_top_N_images(query, df,top_K=15), remove_duplicates(imgs)