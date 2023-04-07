# Image_Search 
Implementation of Image Search Using BLIP and Clip 

In the following project 2, we planned to increase the scope of the details by implementing text image search as an addition to image - image search by CLIP using the COCO2014 dataset.
For project 1 we found the image and text embedding using the CLIP (Contrastive Language-Image Pre-Training) from 100 datasets and predicted the results. 
Here as a second phase we run the model on a COCO dataset with more than 10,000+ images. 
Here we again found the co-similarity between the images and derived the corresponding image based on the given text input in search.

Problem statement:
To do an image search with the help of a pretrained model CLIP (Contrastive Language-Image Pre-Training) and BILP (Blazingly Large Inference Pipeline) form Hugging face.
The goal here is to search and derive the images that are similar with respect to the cos similarity approach.

Dataset:
Here we have used the conceptual_captioning dataset from google to load the images we need.

Implementation of CLIP model:
Here we will utilise the pre trained model from "openai/clip-vit-base-patch32"  which will provide us to encode model with the processor and tokenizer.

Implementation of BLIP model:
The Blip2ForConditionalGeneration class is a specific type of language model that is used for text generation tasks such as summarization, paraphrasing, or question-answering. The from pretrained method is used to load the pre-trained weights of the BLIP2 model into the BLIP model object so that it can be used for text generation tasks.

Implementation of CLIP - Text-to-Image Search:
Created text embedding for the captions and image embedding for images of COCO2014 dataset (10000+ images), and created cosine similarity function to detect similar images for the given input query text

Summary:
In this project we import Google's conceptual captioning data set. Out of this around 3.3 million dataset we gathered limited 100 data and before proceeding we check whether we have a valid data image URL. 
With the help of clip we inherit the pre trained model from "openai/clip-vit-base-patch32" with which we encode the model along with the processor and the tokenizer.

Following we used BLIP through which we did the auto organization from the pre trained model “Salesforce/blip2-opt-2.7b". It does the visual question answering using the model and derives the images that are similar using cos similarity along with the encoded CLIP model.

Reference :
https://www.pinecone.io/learn/clip-image-search/
https://huggingface.co/docs/transformers/main/model_doc/blip-2#transformers.Blip2ForConditionalGeneration.forward.example-2