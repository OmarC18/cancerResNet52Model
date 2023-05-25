
from fastapi import FastAPI, File, UploadFile

import tensorflow as tf

import uuid

app = FastAPI()


model = tf.keras.models.load_model('Model_ResNet52BS128PrecitcionImageSolvedV2.h5')

IMAGE_SIZE = [224, 224]
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpeg"
    contents = await file.read()

    image = tf.image.decode_jpeg(contents, channels=3)
    # convert image to floats in [0, 1] range
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    image = tf.expand_dims(image, axis=0)

    pred = model.predict(image)
    return {"Prediction": pred.tolist()}

