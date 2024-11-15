import streamlit as st
import cv2
import os
import base64
import numpy as np
import requests
import io
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import pillow_heif


load_dotenv()

def main():
    st.set_page_config(page_title="Talk to book", page_icon=":book:")
    st.header("Talk to books :book: ")

    uploaded_pics = st.file_uploader(label="Upload images...", type=["png", "jpg", "jpeg", 'gif', 'heic'], accept_multiple_files=True)
    pic_bytes_list = []

    if uploaded_pics is not None and len(uploaded_pics) > 0:

        ITEM_PER_ROW = 3
        # convert uploaded_pics into a two dimensional list, each row contains ITEM_PER_ROW items
        uploaded_pics = [uploaded_pics[i:i + ITEM_PER_ROW] for i in range(0, len(uploaded_pics), ITEM_PER_ROW)]

        prompt = st.text_area("Prompt", value="These are pictures of a book. Create a script to be used as English learning material, so that child can learn English by reading the script. Use your imagination to cover as many elements in the picutres as possible, trying to create a complete story to chain all pictures together. No extra text is needed in your response, just the script itself.")

        for pics_per_row in uploaded_pics:
            row = st.columns(len(pics_per_row), gap="large", vertical_alignment="bottom")
            for idx, pic in enumerate(pics_per_row):
                if pic.type == 'image/heic':
                    pic = convert_heic_to_jpg(pic.read())
                    pic_bytes = get_image_bytes(pic)
                else:
                    pic_bytes = pic.read()
                pic_bytes_list.append(pic_bytes)
                file_bytes = np.asarray(bytearray(pic_bytes), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)

                row[idx].image(opencv_image, channels="BGR")

    if st.button("Generate story") and uploaded_pics is not None and len(uploaded_pics) > 0:
        with st.spinner("Generating story..."):
            story = pictures_to_story(pic_bytes_list, prompt)
            container = st.container(height=200)
            container.write(story)

            audio_filename, audio_bytes_io = text_to_audio(story)
            st.audio(audio_bytes_io, format='audio/wav')

def to_base64(file_bytes: bytes) -> str:
    b64 = base64.b64encode(file_bytes).decode()
    return f"data:image/png;base64,{b64}"

def pictures_to_story(pic_bytes_list: list[bytes], prompt: str):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
        default_headers={"x-pp-token": os.environ.get("X-PP-TOKEN")},
    )
    print("Prompt: ", prompt)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": 'text',
                    "text": prompt
                },
                *map(lambda byte: {
                    "type": 'image_url',
                    "image_url": {
                        "url": to_base64(byte)
                    }
                }, pic_bytes_list)
            ],
        },
    ]

    result = client.chat.completions.create(
        messages=PROMPT_MESSAGES,
        model="gpt-4o-mini",
        max_tokens=500,
    )
    return result.choices[0].message.content

def convert_heic_to_jpg(heic_file: bytes) -> Image.Image:
    heif_image = pillow_heif.read_heif(heic_file)
    image = Image.frombytes(
        heif_image.mode,
        heif_image.size,
        heif_image.data,
        "raw",
        heif_image.mode,
        heif_image.stride,
    )
    return image

def get_image_bytes(image: Image.Image, format: str = 'JPEG') -> bytes:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=format)
    return byte_arr.getvalue()

def text_to_audio(text):
    response = requests.post(
        f"{os.environ['OPENAI_API_BASE']}audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "x-pp-token": os.environ.get("X-PP-TOKEN"),
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": 'alloy',
        },
    )

    if response.status_code != 200:
        raise Exception("Request failed with status code")
    # ...
    # Create an in-memory bytes buffer
    audio_bytes_io = io.BytesIO()

    # Write audio data to the in-memory bytes buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io

if __name__ == "__main__":
    main()