import pickle
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
import uvicorn
# import mysql.connector
# from scipy.sparse import coo_matrix, vstack, hstack
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.cluster import KMeans
import pickle
from gensim.models import Word2Vec, Phrases
from pyvi import ViPosTagger, ViTokenizer #Tokenize tiếng Việt
import gensim
from gensim.models import Word2Vec
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import *

dictionary = {
    "👹": 'negative', "👻": 'positive', "💃": 'positive','🤙': 'positive', '👍': 'positive',
    "💄": 'positive', "💎": 'positive', "💩": 'positive',"😕": 'negative', "😱": 'negative', "😸": 'positive',
    "😾": 'negative', "🚫": 'negative',  "🤬": 'negative',"🧚": 'positive', "🧡": 'positive','🐶': 'positive',
    '👎': 'negative', '😣': 'negative','✨': 'positive', '❣': 'positive','☀': 'positive',
    '♥': 'positive', '🤩': 'positive', 'like': 'positive', '💌': 'positive',
    '🤣': 'positive', '🖤': 'positive', '🤤': 'positive', '😢': 'negative',
    '❤': 'positive', '😍': 'positive', '😘': 'positive', '😪': 'negative', '😊': 'positive',
    '😁': 'positive', '💖': 'positive', '😟': 'negative', '😭': 'negative',
    '💯': 'positive', '💗': 'positive', '♡': 'positive', '💜': 'positive', '🤗': 'positive',
    '😨': 'negative', '☺': 'positive', '💋': 'positive', '👌': 'positive',
    '😖': 'negative', '😀': 'positive', '😡': 'negative', 
    '😠': 'negative', '😒': 'negative', '🙂': 'positive', '😏': 'negative', '😝': 'positive', 
    '😙': 'positive', '😤': 'negative', '😎': 'positive', '😆': 'positive', '💚': 'positive',
    '✌': 'positive', '💕': 'positive', '😞': 'negative', '😓': 'negative', '️🆗️': 'positive',
    '😉': 'positive', '😂': 'positive', ':v': 'positive', '😋': 'positive',
    '💓': 'positive', '😐': 'negative', ':3': 'positive', '😫': 'negative', '😥': 'negative',
    '😃': 'positive', '😬': 'negative', '😌': 'negative', '💛': 'positive', '🤝': 'positive', '🎈': 'positive',
    '😗': 'positive', '🤔': 'negative', '😑': 'negative', '🔥': 'negative', '🙏': 'negative',
    '🆗': 'positive', '😻': 'positive', '💙': 'positive', '💟': 'positive',
    '😚': 'positive', '❌': 'negative', '👏': 'positive',
    '🌝': 'positive',  '🌷': 'positive', '🌸': 'positive', '🌺': ' positive ',
    '🌼': 'positive', '🍓': 'positive', '🐅': 'positive', '🐾': 'positive', '👉': 'positive',
    '💐': 'positive', '💞': 'positive', '💥': 'positive', '💪': 'positive', '🎉': 'positive',
    '💰': 'positive',  '😇': 'positive', '😛': 'positive', '😜': 'positive', '😄': 'positive',
    '🙃': 'positive', '🤑': 'positive', '🤪': 'positive','☹': 'negative',  '💀': 'negative',
    '😔': 'positive', '😧': 'negative', '😩': 'negative', '😰': 'negative', '😳': 'negative',
    '😵': 'positive', '😶': 'negative', '🙁': 'negative',
    # 
    u'ô_kêi': 'ok', 'okie': 'ok', u'ô_kê': 'ok', 'okey': 'ok', u'ôkê': 'ok', 'oki': 'ok', 'oke': 'ok', 
    'okay': 'ok', 'tks': u'cám_ơn', 'thks': u'cám_ơn', 'thanks': u'cám_ơn', 'ths': u'cám_ơn', 'thank': u'cám_ơn', 
    u'cảm_ơn': u'cám_ơn', '⭐': 'star', '*': 'star','🌟': 'star', 'kg': u'không', 'not': u'không', 
    'k': u'không', 'kh': u'không', u'kô': u'không', 'hok': u'không', 'kp': u'không_phải', 'ko': u'không', 
    'khong': u'không', 'he_he': 'positive','hehe': 'positive', 'hihi': 'positive', 'haha': 'positive', 
    'hjhj': 'positive', 'lol': 'negative','cc': 'negative', 'cute': u'dễ_thương', 'huhu': 'negative', 'vs': u'với',
    'wa': u'quá', 'wá': u'quá', 'j': u'gì', 'sz': u'kích_cỡ', ' size': u'kích_cỡ', 'đx': u'được', 'dk': u'được',
    'dc': u'được', 'đk': u'được', 'đc': u'được','authentic': u'chính_hãng','aut': u'chính_hãng', 'auth': u'chính_hãng', 
    'thick': u'dày', 'store': u'cửa_hàng', 'shop': u'cửa_hàng', 'sp': u'sản_phẩm', 'gud': u'tốt','weldone':'tốt', 
    'good': u'tốt', 'very': u'rất', u'gút': u'tốt', 'gut': u'tốt', u'tot': u'tốt', 'nice': 'tốt', 'perfect': u'hoàn_hảo', 
    'bt': u'bình_thường', 'time': u'thời_gian', u'qá': u'quá', 'ship': u'giao_hàng', 'm': u'mình', 'mik': u'mình', 
    'product': u'sản_phẩm', 'quality': u'chất_lượng','chat': u'chất', 'excelent': u'hoàn_hảo', 'bad': u'tệ',
    'fresh': u'tươi', 'sad': u'buồn', 'date': u'hạn_sử_dụng', 'hsd': u'hạn_sử_dụng','quickly': u'nhanh', 
    'quick': u'nhanh', 'fast': u'nhanh', 'delivery': u'giao_hàng', u'síp': u'giao_hàng', 'shipper': u'người_giao_hàng', 
    'beautiful': u'đẹp', 'tl': u'trả_lời', 'r': u'rồi', 'shopE': u'cửa_hàng', 'order': u'đặt_hàng', 
    u'chất_lg': u'chất_lượng', 'sd': u'sử_dụng', 'dt': u'điện_thoại', 'nt': u'nhắn_tin', u'sài': u'xài', 
    'bjo': u'bao_giờ', 'thik': u'thích', 'sop': u'cửa_hàng', 'fb': 'facebook','dep': u'đẹp', 'xau': u'xấu', 
    'delicious': u'ngon', u'hàg': u'hàng', u'qủa': u'quả', 'iu': u'yêu', 'fake': u'giả_mạo', 'trl': u'trả_lời', 
    'por': u'tệ', 'poor': u'tệ', 'ib': u'nhắn_tin', 'rep': u'trả_lời', u'fback': 'feedback', 'fedback': 'feedback',
    'mn': u'mọi_người', 'cx': u'cũng', '&': u'và'
}

stop_words = np.array([u'đã', u'sẽ', u'không', u'nhưng', u'mỗi', u'chỉ', u'vì', u'bởi', u'bởi_vì', u'của', u'khi', 
                       u'ai', u'thì', u'là', u'có_lẽ', u'có_vẻ', u'nhưng_mà', u'tuy_nhiên', u'với', u'và', u'bây_giờ', 
                       u'có', u'vài', u'ít', u'lần_nữa', u'một_lần_nữa', u'nào', u'duy_nhất', u'cho', 
                       u'đây', u'đó', u'kia', u'kìa', u'này', u'thôi', u'cho_đến_khi', u'nên', u'hoặc', u'mà', 
                       u'chẳng', u'mình', u'tôi', u'nữa', u'tại', u'kia', u'rằng', u'bây_giờ', u'được', u'bị', 
                       u'sau_đó', u'trong_lúc', u'trong_khi', u'cái_gì', u'đấy', u'đó', u'về', u'còn', u'hầu_hết',
                       u'cũng', u'mặc_dù', u'luôn', u'một', u'bất_kỳ', u'bất_kì', u'nữa', u'ở', u'trở_nên', u'có_thể',
                       u'có_lẽ', u'hơn_nữa', u'thậm_chí', u'vì_vậy', u'rất', u'quá', u'hơi_hơi', u'tại_sao', u'lắm',
                       u'thấy', u'nhiều', u'thực_sự', u'lâu_lâu', u'đâu', u'để'])

class Sentence(BaseModel):
    sen: str

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def transform(txt):
    txt = txt.lower()
    txt =  txt.replace(':3','')
    txt =  txt.replace(':v','')
    txt =  txt.replace(':o','')
    txt =  txt.replace('"','')
    txt = re.sub("[,#.:@;()$`~/-\^<>+=!]", '', txt)
    return txt

def tokenize(txt):
    return ViPosTagger.postagging(ViTokenizer.tokenize(txt))[0]

def transform_abbreviation(txt):
    rev_subs = {k:v for k,v in dictionary.items()}
    return [rev_subs.get(item,item)  for item in txt]

def remove_stopwords(input_text):
    whitelist = ["không", "không_thể", "chẳng"]
    # words = input_text.split() 
    clean_words = [word for word in input_text if (word not in stop_words or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words)

w2v_model = pickle.load(open("w2v_model",'rb'))
bow_vectorizer = pickle.load(open("bow_vectorizer",'rb'))

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            # print(w2v_model.wv[word].reshape((1, size)))
            vec += w2v_model.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            print("ERROR while word to vector")
            continue
    if count != 0:
        vec /= count
    return vec

lor_th = 0.802978
rf_th = 0.731134
gb_th = 0.872183
pb_th = 0.988472

lor_model = pickle.load(open('logistic.sav','rb'))
rf_model = pickle.load(open('RF.sav','rb'))
gb_model = pickle.load(open('GB.sav','rb'))
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("bdh240901/phoBert-BigData")

def pb_predict_func(text):
    text = segment_text((remove_any_special_left(remove_number_characters(remove_special_characters(text)))))
    model.to("cpu")
    input_values = tokenizer(text,return_tensors="pt",truncation=True).to("cpu")
    with torch.no_grad():
        logits = model(**input_values).logits
    # batch['label_id'] = torch.argmax(logits, dim=-1)
    # return batch
    return softmax(logits.numpy())[:,1]

# print(model.predict(word_vector(remove_stopwords(transform_abbreviation(tokenize(transform("cuộc hành_trình ngàn dặm phải bắt_đầu bằng bước_đi nhỏ_bé đầu_tiên_cảm_ơn bộ sách dịch happylive")))),300)))
# print(lor_model.predict_proba(bow_vectorizer.transform([remove_stopwords(transform_abbreviation(tokenize(transform("sản phẩm này ngon"))))]))[0,1] >= 0.880753)

# print(rf_model.predict_proba(bow_vectorizer.transform([remove_stopwords(transform_abbreviation(tokenize(transform("sản phẩm này ngon"))))]))[0,1] >= 0.880753)

# print(gb_model.predict_proba(bow_vectorizer.transform([remove_stopwords(transform_abbreviation(tokenize(transform("sản phẩm này ngon"))))]))[0,1] >= 0.933088)


# app.mount("/static", StaticFiles(directory="static"), name="static")

class Output(BaseModel):
    LOR: str
    RF: str
    GB: str
    PB: str

@app.post('/predict', response_model=Output)
async def predict(data : Sentence):
    gb_data = data.sen
    data = data.sen
    data = bow_vectorizer.transform([remove_stopwords(transform_abbreviation(tokenize(transform(data))))])
    lor_predict = "Positive" if lor_model.predict_proba(data)[0,1] >= lor_th else "Negative"
    rf_predict = "Positive" if rf_model.predict_proba(data)[0,1] >= rf_th else "Negative"
    gb_predict = "Positive" if gb_model.predict_proba(data)[0,1] >= gb_th else "Negative"
    pb_predict = "Positive" if pb_predict_func(gb_data)[0] >= gb_th else "Negative"
    return {'LOR': lor_predict, 'RF': rf_predict, 'GB': gb_predict, 'PB': pb_predict}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)
