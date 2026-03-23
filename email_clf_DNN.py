import kagglehub
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers


path = Path(kagglehub.dataset_download("chandramoulinaidu/spam-classification-for-basic-nlp"))
data = pd.read_csv(path / "Spam Email raw text for NLP.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=1, stratify=data['CATEGORY'])
y_train = train_data['CATEGORY'].to_numpy().astype(int)
y_test = test_data['CATEGORY'].to_numpy().astype(int)

vocab_size = 10000
max_len = 150

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['MESSAGE'])

train_seq = tokenizer.texts_to_sequences(train_data['MESSAGE'])
train_padd = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
test_seq = tokenizer.texts_to_sequences(test_data['MESSAGE'])
test_padd = pad_sequences(test_seq, maxlen=max_len, padding="post", truncating="post")

model = tf.keras.Sequential([
	layers.Embedding(vocab_size, 32),
	layers.GlobalAveragePooling1D(),
	layers.Dense(24, activation="relu"),
	layers.Dropout(0.5),
	layers.Dense(1, activation="sigmoid")
	])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', 'precision', 'recall'])
model.fit(
	train_padd,
	y_train,
	epochs=10,
	batch_size=32,
	validation_data=(test_padd, y_test),
	verbose=2)
