# Citation info : 
'''
@misc{https://doi.org/10.48550/arxiv.2201.12086,
  doi = {10.48550/ARXIV.2201.12086},
  url = {https://arxiv.org/abs/2201.12086},
  author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
'''

# This is a Streamlit app that takes an image as input and generates captions for the image using a pre-trained image captioning model from the Hugging Face Transformers library.

# load required packages
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

# Load the pre-trained model from Transformers.
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# The model is cached using the @st.cache_resource decorator so that it does not need to be loaded each time the app is run.
@st.cache_resource
def create_model():
    # The BlipForConditionalGeneration model is a transformer-based language model that is used for image captioning.
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return model

def create_caption(raw_image,num):
    # Convert the image to RGB format if necessary.
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert(mode="RGB")
    # Process the image using the BlipProcessor from Transformers.
    inputs = processor(raw_image, return_tensors="pt")
    # Load the pre-trained captioning model using the create_model function.
    model = create_model()
    # Generate captions using the model.
    out = model.generate(**inputs, num_beams=num, num_return_sequences=num, max_new_tokens=20)
    # Decode the generated captions and append them to a list.
    output = []
    for caption in out:
        output.append(processor.decode(caption, skip_special_tokens=True))
    return output  # Return the list of captions.

def main():
    st.title('Caption Creater')
    # Allow the user to select an image either by providing a URL or by uploading a file.
    option = st.radio('Select an option:', ('Enter an image URL','Upload an image'))
    if option=="Enter an image URL":
        url = st.text_input('Enter the URL of an image:')
        if url:
            try:           
                response = requests.get(url)
                raw_image = Image.open(BytesIO(response.content))
                st.image(raw_image)
            except:
                st.write('Error: Invalid URL or unable to download image.')
    else:
        uploaded_file = st.file_uploader("Choose a file", type=["jpg","png"])
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            st.image(raw_image)
    # Ask the user how many captions they want to generate.
    num = st.number_input('Number of captions: ',1,5)
    # When the user clicks the "Predict" button, generate the captions using the create_caption function and display them on the page.
    if st.button('Predict'):
        output = create_caption(raw_image,num)
        for i in range(num):
            st.write('Caption ',str(i+1),': ',output[i])

if __name__ == '__main__':
    main()