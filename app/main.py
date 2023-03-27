from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import requests
# from model import tokenizer, MAX_LENGTH, PADDING_TYPE, TRUNC_TYPE
import pandas as pd

data = pd.read_csv('/app/final.csv').drop(columns=['Unnamed: 0'])
#data = pd.read_csv('./final.csv').drop(columns=['Unnamed: 0'])

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(data['message'])

app = FastAPI()
app = FastAPI(ssl_keyfile="/app/privatekey.pem", ssl_certfile="/app/certificate.pem")


model = load_model('/app/model.h5')
#model = load_model('./model.h5')


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


class email_spam_detection(BaseModel):
    email_content: str


def prepare_sentence(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    return padded


@app.post("/")
async def predict(item: email_spam_detection):
    sentence = prepare_sentence(item.email_content)
    prediction = model.predict(sentence)
    print(prediction)
    result = f'email is spam' if prediction > 0.5 else f'email is not spam'
    is_spam = True if prediction > 0.5 else False
    confidence = float(prediction[0][0])
#    url_data = requests.get('http://localhost:5000/', json={'email_content': item.email_content})
#    url_data = url_data.json()
    return {"prediction": result,
            "is_spam": is_spam,
            "confidence": confidence}
