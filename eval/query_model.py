from openai import OpenAI
import google.generativeai as genai

import base64
import os
import time

client = OpenAI(
   api_key=os.environ.get('OPENAI_API_KEY'),
   base_url=os.environ.get('OPENAI_API_URL')
   )

def query_OpenAI(images_path, prompt, retry=10, model_name="gpt-4o"):
    """
    Query OpenAI model with the prompt and a list of image paths. The temperature is set to 0.0 and retry is set to 10 if fails as default setting.

    Parameters:
    - images: PIL.JpegImagePlugin.JpegImageFile, the image.
    - prompt: String, the prompt.
    - retry: Integer, the number of retries.
    """

    base64_images = []
    for image_path in images_path:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            base64_images.append(encoded_image)

    for r in range(retry):
        print(r)
        try:
            input_dicts = [{"type": "text", "text": prompt}]
            for i, image in enumerate(base64_images):
                input_dicts.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "low"}})
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                    "role": "user",
                    "content": input_dicts,
                    }
                ],
                max_tokens=1024,
                n=1,
                temperature=0.0,
            )
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(1)
    return f'Failed: Query {model_name} Error'