import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import requests
from googletrans import Translator
from datetime import datetime

translator = Translator()

URL = "Украина"#Что ищем


#Size - Кол-во статей
#В браузере можно это вставить и получишь весь массив: https://lenta.ru/search/v2/process?query=Украина&from=0&size=20&sort=2&title_only=0&domain=1&modified%2Cformat=yyyy-MM-dd
URL_Lenta = (f"https://lenta.ru/search/v2/process?query={URL}&from=0&size=20&sort=2&title_only=0&domain=1&modified%2Cformat=yyyy-MM-dd")

requestLenta = requests.get(URL_Lenta)

responseLenta = requestLenta.json()

responseEng = [] #Статьи на англе
responseRus = [] #Статьи на русском

for i in range(len(responseLenta["matches"])):
    main = responseLenta["matches"][i]["text"]

    responseEng.append(
        translator.translate(text=str(main), src='ru', dest='en').text)
    responseRus.append({
        "num": i+1,
        "main": main,
        "url": responseLenta["matches"][i]["url"],
        "time": datetime.fromtimestamp(responseLenta["matches"][i]["lastmodtime"]).strftime("%d.%m.%y"),
        "key": ''
    })

tokenizer = Tokenizer()

model = load_model('news_predict.h5')
X = responseEng
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=1000)
(model.predict(X) >= 0.5).astype(int)