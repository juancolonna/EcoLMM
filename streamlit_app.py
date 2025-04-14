from openai import OpenAI
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from google import genai
import json

st.set_page_config(layout="wide")

st.logo("logo.png")
# Set page config for full width
st.title("EcoLMM Image descrition")

# Initialize session state for API keys if not exists
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""

if "species_list" not in st.session_state:
    st.session_state.species_list = ["Crax globulosa","Didelphis albiventris","Leopardus wiedii", "Panthera onca", "Sapajus macrocephalus", "Sciurus spadiceus", "Tupinambis teguixin"]

# Sidebar configuration
with st.sidebar:
    #st.title("Configuration")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key here",
    )
    st.session_state.openai_api_key = openai_api_key
    
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Gemini API key here",
    )
    st.session_state.gemini_api_key = gemini_api_key
    
    # Model selection
    st.subheader("Model Selection")
    model_type = st.radio(
        "Select API Provider",
        ["OpenAI", "Gemini"],
        key="provider_radio"
    )

    # Model list selection based on provider
    if model_type == "OpenAI":
        selected_model = st.selectbox(
            "Select OpenAI Model",
            ["gpt-4o-mini", "gpt-4-turbo"],
            index=0,
            help="Choose the OpenAI model to use for analysis"
        )
        st.session_state.selected_model = selected_model

    else:  # Gemini
        selected_model = st.selectbox(
            "Select Gemini Model",
            ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro-exp-03-25", "gemini-1.5-flash"],
            index=0,
            help="Choose the Gemini model to use for analysis",
        )
        st.session_state.selected_model = selected_model

    # Initialize clients based on selected model
    if model_type == "OpenAI" and st.session_state.openai_api_key:
        client = OpenAI(api_key=st.session_state.openai_api_key)
    
    if model_type == "Gemini" and st.session_state.gemini_api_key:
        client = genai.Client(api_key=st.session_state.gemini_api_key)
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Controls randomness. Lower values are more focused, higher values more creative."# Initialize clients based on selected model
    )

    # Keywords for image analysis
    keywords = st.multiselect(
        "Select species to focus the analysis",
        st.session_state.species_list,
        default=st.session_state.species_list,
        help="Select species to guide the image analysis"
    )

    # Species list management
    new_species = st.text_input(
        "Add new species (scientific name) to the list",
        help="Enter a species name and press Enter to add it to the list",
    )
    
    if new_species and new_species not in st.session_state.species_list:
        st.session_state.species_list.append(new_species)
        st.success(f"Added {new_species} to the list")
        st.rerun()


def decode_json(response):
    # Try to parse the response as JSON and display it nicely
    try:
        # Clean the response text to ensure it's valid JSON
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        json_data = json.loads(json_str)
        st.json(json_data)
    except json.JSONDecodeError:
        # If JSON parsing fails, show the raw response
        st.write("Raw response (not valid JSON):")
        st.write(response)

species_str = ", ".join(st.session_state.species_list)

prompt = f"""You are an expert in image recognition and wildlife biology. You will be provided with a camera trap image, and your task is to analyze it thoroughly. 
Describe the image in detail, including the identification of any animals present, the species names, the common names, and the number of individuals.
Consider the following species as possible candidates: {species_str}, but do not limit your analysis solely to these species.

Return your analysis in JSON format with the following fields:
- "Detection": "Yes" if animals are detected, otherwise "None".
- "Species Name": The scientific name of the species.
- "Species common name": The commonly used name.
- "Number of individuals": The count of the animals detected.
- "Image description": A detailed narrative of what is visible in the image.

If no animals are present, return a JSON object where all fields are populated with "None".
"""

# Function to encode the image
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

ready = False

# Main content area
st.write("")  # Add some space before the columns

# Create columns with proper spacing and reduced gap
col1, col2 = st.columns([1, 2], gap="small")

with col1:
    # File uploader
    img_file_buffer = st.file_uploader(
        'Upload a PNG, JPG or JPEG camera trap image',
        type=['png','jpg','jpeg'],
        help="Upload an image to analyze its contents"
    )
    
    if img_file_buffer is not None:
        # Open and process the image
        image = Image.open(img_file_buffer)
        # Resize while maintaining aspect ratio
        max_size = (640, 640)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        # Display the uploaded image in the first column
        st.image(image, use_container_width=True)
        ready = True
    
# Get response based on selected model
with col2:
    if model_type == "OpenAI" and st.session_state.openai_api_key and ready:
        # Encode the image
        base64_image = encode_image(image)
        
        # Create the message for the API
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        },
                    },
                ],
            }
        ]
        
        # Get response from OpenAI
        st.write('model:', st.session_state.selected_model) 
        response = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=messages,
            temperature=temperature
        )
        decode_json(response.choices[0].message.content)
        ready = False
        
    elif model_type == "Gemini" and st.session_state.gemini_api_key and ready:
        # Configure generation config for Gemini
        generation_config = genai.types.GenerateContentConfig(
            temperature=temperature,
        )
        st.write('model:', st.session_state.selected_model)
        response = client.models.generate_content(
            model=st.session_state.selected_model,
            contents=[prompt, image],
            config=generation_config
        )
        decode_json(response.text)
        ready = False

    else:
        st.write("Please enter your API keys in the sidebar and upload an image")
