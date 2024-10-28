import streamlit as st
import pandas as pd
import numpy as np
from itables.streamlit import interactive_table
import pyarrow
from streamlit.components.v1 import html
from streamlit.components.v1.components import MarshallComponentException
from PIL import Image
from streamlit_navigation_bar import st_navbar
import pages as pg
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
import os
from google.cloud import storage
from vertexai.vision_models import MultiModalEmbeddingModel


PROJECT_ID = os.environ.get("bagchi-genai-bb")
LOCATION = os.environ.get("us_central1")
BUCKET = "bagchi-genai-bb"
BUCKET_URI = f"gs://{BUCKET}/"


vertexai.init(project=PROJECT_ID, location=LOCATION)
print(f"Using vertexai version: {vertexai.__version__}")
# storage_client = storage.Client()
# bucket = storage_client.get_bucket(BUCKET)

embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
print(f"Using embedding_model: {embedding_model}")


@st.cache_resource
def load_models() -> tuple[GenerativeModel, GenerativeModel]:
    """Load Gemini 1.5 Flash and Pro models."""
    return GenerativeModel("gemini-1.5-flash"), GenerativeModel("gemini-1.5-pro")

gemini_15_flash, gemini_15_pro = load_models()
print(f"Models loaded: {gemini_15_flash}")

def get_gemini_response(
    model: GenerativeModel,
    contents: list,
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1, max_output_tokens=2048
    ),
    stream: bool = True,
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )
    print(f"Response: {responses}")

    if not stream:
        return responses.text

    final_response = []
    for r in responses:
        try:
            final_response.append(r.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


def get_model_name(model: GenerativeModel) -> str:
    """Get Gemini Model Name"""
    model_name = model._model_name.replace(  # pylint: disable=protected-access
        "publishers/google/models/", ""
    )
    return f"`{model_name}`"


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def generate_story(text_gen_prompt: str) -> str:
    print(f"calling gemini response: {gemini_15_flash}")

    temperature = 0.30
    max_output_tokens = 2048
    
    config = GenerationConfig(
    temperature=temperature, max_output_tokens=max_output_tokens
    )

    response =  get_gemini_response(
    gemini_15_flash,  # Use the selected model
    text_gen_prompt,
    generation_config=config,
    )
    print(f"received gemini response: {response}")

    return response

def generate_image_classification(images_content) -> str:
    print(f"calling gemini response: {gemini_15_flash}")

    temperature = 0.30
    max_output_tokens = 2048
    
    config = GenerationConfig(
    temperature=temperature, max_output_tokens=max_output_tokens
    )

    response =  get_gemini_response(
    gemini_15_flash,  # Use the selected model
    images_content,
    generation_config=config,
    )
    print(f"received gemini response: {response}")

    return response
