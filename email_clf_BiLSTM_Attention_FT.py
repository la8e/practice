import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import kagglehub as kgl
import get_embeddings as ge

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Layer)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


VACAB_S = 20000
MAX_LEN = 200
EMBEDDING_DIM = 300 # updated according to fasttext dims

path = Path(kgl.dataset_download(""))
data = pd.read_csv(path / "")

def clean_text(text):
	text = str(text).lower()
	text = re.sub(r"http\S+|www\S+", " URL ", text)
	text = re.sub(r"\+?\d[\d\s\-]{7,}\d", " NUM ", text)
	text = re.sub(r"<.*?>", "", text)
	text = re.sub(r"[^a-zA-Z]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text

data["clean_text"] = data["text"].apply(clean_text)
print(data.shape)
data["clean_text"] = data["clean_text"].replace("", np.nan)
data = data.dropna()

train_data, test_data = train_test_split(data,
										test_size=0.2,
										random_state=42,
										stratify=data["label"])
y_train = train_data["label"].to_numpy().astype(int)
y_test = test_data["label"].to_numpy().astype(int)

def get_padded_seq(text, tokenizer):
	seq = tokenizer.texts_to_sequences(text)
	padd_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
	return padd_seq

tokenizer = Tokenizer(num_words=VACAB_S, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data["clean_text"])
train_padded = get_padded_seq(train_data["clean_text"], tokenizer)
test_padded = get_padded_seq(test_data["clean_text"], tokenizer)

embeddings_index = {}
embedding_matrix = ge.get_embedding_matrix("fasttext-path-here", tokenizer, VOCAB_S)

class Attention(Layer):
	def __init__(self, **kwargs):
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape): # (batch, timesteps, features)
		self.W = self.add_weight(name="atten_weight",
								shape=(input_shape[-1], 1),
								initializer="glorot_uniform",
								trainable=True)
		self.b = self.add_weight(name="atten_bias",
								shape=(input_shape[1], 1),
								initializer="zeros",
								trainable=True)
		super(Attention, self).build(input_shape)

	def call(self, x):
		e = tf.keras.backend.tanh(th.keras.backend.dot(x, self.W) + self.b)
		a = tf.keras.backend.softmax(e, axis=1)
		weighted = x * a
		context_vec = tf.keras.backend.sum(weighted, axis=1)
		return context_vec


inputs = Input(shape=(MAX_LEN,))
embedd_layer = Embedding(VOCAB_S, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(inputs)
biLSTM_layer = Bidirectional(LSTM(64, return_sequences=True))(embedd_layer)
attention = Attention(biLSTM_layer)
dropout = Dropout(0.5)(attention) # prevent overfitting
outputs = Dense(1, activation="sigmoid")(dropout)

model = Model(inputs, outputs)
model.compile(optimizer="adam",
			  loss="binary_crossentropy",
			  metrics=["accuracy", "precision", "recall"])

compute_weights = class_weight.compute_class_weights("balanced",
					classes=np.unique(y_train),
					y=y_train)
class_weights = dict(enumerate(compute_weights))


clf = model.fit(train_padded,
				y_train,
				epochs=10,
				batch_size=64,
				validation_split=0.1,
				class_weight=class_weights)