import os
import fasttext as ft
import numpy as np

# path
EMBEDDS_PATH = "embedds.npy"

def get_embedding_matrix(fasttext_path, tokenizer, vocab_size):
	if os.path.exists(EMBEDDS_PATH):
		print(f"[*] loading cached matrix from {EMBEDDS_PATH}...")
		return np.load(EMBEDDS_PATH)

	if not os.path.exists(fasttext_path):
		raise FileNotFoundError(f"! You need the FastText .bin file at {fasttext_path}")

	ft_model = ft.load_model(fasttext_path)
	dim = ft_model.get_dimension()

	matrix = np.zeros((vocab_size, dim))

	for word, i in tokenizer.word_index.items():
		if i < vocab_size:
			matrix[i] = ft_model.get_word_vector(word)

	print(f"[*] Saving matrix to {EMBEDDS_PATH} for future runs...")
	np.save(EMBEDDS_PATH, matrix)

	del ft_model # free RAM
	return matrix