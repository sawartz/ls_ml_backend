from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase



import os
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='./key.json'
client = vision.ImageAnnotatorClient()
image = vision.Image()

def detect_text(path,original_width,original_height):
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    js_data = []
    n=1
    for text in texts[1:]:
        text_description = text.description

        vertices = [
            [vertex.x,vertex.y] for vertex in text.bounding_poly.vertices
        ]
        min_x = min(vertices, key=lambda v: v[0])[0]
        min_y = min(vertices, key=lambda v: v[1])[1]
        max_x = max(vertices, key=lambda v: v[0])[0]
        max_y = max(vertices, key=lambda v: v[1])[1]
        x = min_x*100/original_width
        y = min_y*100/original_height
        width = (max_x - min_x)*100/original_width
        height = (max_y - min_y)*100/original_height
        dict_1= {
            "original_width": original_width,
            "original_height": original_height,
            "image_rotation": 0,
            "value": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "rotation": 0,
                "labels": [
                    "Text"
                ]
            },
            "id": "bb"+str(n),
            "from_name": "label",
            "to_name": "image",
            "type": "labels"
        }
        dict_2 ={
            "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                   "text": [
                      text_description
                   ]
                },
                "id": 'bb'+str(n),
                "from_name": "transcription",
                "to_name": "image",
                "type": "textarea"
        }
        js_data.append(dict_1)
        js_data.append(dict_2)
        n+=1

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return js_data


import urllib.request
from PIL import Image
class NewModel(LabelStudioMLBase):
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        for task in tasks:
            img_url = 'http://localhost:8081'+task['data']['ocr']
            filename = img_url.split("/")[-1]
            # image downloading from label-studio
            #####################################
            opener = urllib.request.build_opener()
            opener.addheaders = [('Authorization', 'Token ad1d61e52e859f2d2a3a136e1615f9127a4491ea')]
            urllib.request.install_opener(opener)
            image = urllib.request.urlretrieve(img_url, filename)
            #####################################
            image = Image.open('./' + filename)
            width, height = image.size
            image.close()
            path = './' + filename
            result = detect_text(path,width,height)
            return [{"result": result}]
            