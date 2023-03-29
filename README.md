# Image_Search 
Implementation of Image Search Using BLIP and Clip 

Problem statement:
To do an image search with the help of a pretrained model CLIP (Contrastive Language-Image Pre-Training) and BILP (Blazingly Large Inference Pipeline) form Hugging face.
The goal here is to search and derive the images that are similar with respect to the cos similarity approach.

Dataset:
Here we have used the conceptual_captioning dataset from google to load the images we need.

Implementation of CLIP model:
Here we will utilise the pre trained model from "openai/clip-vit-base-patch32"  which will provide us to encode model with the processor and tokenizer.

Implementation of BLIP model:
The Blip2ForConditionalGeneration class is a specific type of language model that is used for text generation tasks such as summarization, paraphrasing, or question-answering. The from pretrained method is used to load the pre-trained weights of the BLIP2 model into the BLIP model object so that it can be used for text generation tasks.

Summary:
In this project we import Google's conceptual captioning data set. Out of this around 3.3 million dataset we gathered limited 100 data and before proceeding we check whether we have a valid data image URL. 
With the help of clip we inherit the pre trained model from "openai/clip-vit-base-patch32" with which we encode the model along with the processor and the tokenizer.

Following we used BLIP through which we did the auto organization from the pre trained model “Salesforce/blip2-opt-2.7b". It does the visual question answering using the model and derives the images that are similar using cos similarity along with the encoded CLIP model.

Reference :
https://www.pinecone.io/learn/clip-image-search/
https://huggingface.co/docs/transformers/main/model_doc/blip-2#transformers.Blip2ForConditionalGeneration.forward.example-2