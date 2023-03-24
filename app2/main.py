import torch
import torch.nn as nn
import pandas as pd
import re

########## Data Preprocessing #######################
#dataset = pd.read_csv('malicious_phish.csv')
dataset = pd.read_csv('/app/malicious_phish.csv')

class Label_Index:
    def __init__(self, dataset):
        self.labels = dataset['type'].unique()
        self.label_index = {label: index for index, label in enumerate(self.labels)}
        self.index_label = {index: label for index, label in enumerate(self.labels)}
    
    def indexes_labels(self, dataset):
        return dataset['type'].map(self.index_label)

    def labels_indexes(self, dataset):
        return dataset['type'].map(self.label_index)
    def __call__(self, label):
        return self.label_index[label]

label_index = Label_Index(dataset)
label_index('phishing')


class Char_Index:
    def __init__(self, urls) -> None:
        self.char_index = {}
        self.index_char = {}
        for url in urls:
            for char in url:
                if char not in self.char_index:
                    self.char_index[char] = len(self.char_index)
                    self.index_char[len(self.index_char)] = char
    
    def string_indexes(self, string):
        return [self.char_index[char] for char in string]

char_index = Char_Index(dataset['url'])
char_index.string_indexes(dataset.url[0]), len(char_index.char_index)


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(GRU, self).__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)  
        if self.bidirectional == True:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) 
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        out, _ = self.gru(x, h0)    
        if self.bidirectional == True:
            out = out[:, -1, :self.hidden_size] + out[:, 0, self.hidden_size:] 
        return self.fc(out)
gru_model = GRU(len(char_index.char_index), 128, 128, len(label_index.labels), bidirectional=True, num_layers=1)
gru_model.load_state_dict(torch.load('gru_model.pth', map_location=torch.device('cpu')))

########## Prediction #########################
    

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


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

def predict(model, url):
    model.eval()
    url = torch.tensor(char_index.string_indexes(url))
    url = url.unsqueeze(0)
    with torch.no_grad():
        output = model(url)
        output = torch.argmax(output, dim=1)
        return label_index.index_label[output.item()]
    

class url_spam_detection(BaseModel):
    email_content: str


def find_url(message):
    url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    www_regex = re.compile(r'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    urls = re.findall(url_regex, message)
    www = re.findall(www_regex, message)
    urls.extend(www)
    return urls


@app.get("/")
async def url_predict(item: url_spam_detection):
    urls = find_url(item.email_content)
    result = []
    for url in urls:
        result.append([url, predict(gru_model, url)])

    return result