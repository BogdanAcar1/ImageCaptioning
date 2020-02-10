from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import argmax
from dataset import Dataset
import os

class CaptionGenerator():
	def __init__(self, model, vocabulary):
		self.start_token = 'starttoken'
		self.end_token = 'endtoken'
		self.model = load_model(model)
		self.vocabulary = vocabulary
		self.image_path = 'Flicker8k_Dataset'

	def _index_to_word(integer):
		return self.vocabulary.index_to_word(integer)

	def generate_caption(self, img_features):
		caption = self.start_token
		for i in range(self.vocabulary.caption_max_length):
			sequence = self.vocabulary.tokenizer.texts_to_sequences([caption])[0]
			sequence = pad_sequences(
				[sequence],
				maxlen = 32 #self.vocabulary.caption_max_length
			)
			predicted = argmax(self.model.predict([img_features, sequence], verbose = 0))
			word = self.vocabulary.index_to_word.get(predicted, None)
			if word is None or word == self.end_token:
				break
			caption += ' ' + word
		return caption

	def display_image_caption(self, image_id, image_features):
		caption = self.generate_caption(image_features)
		img = mpimg.imread(os.path.join(self.image_path, f'{image_id}.jpg'))
		plt.imshow(img)
		plt.title(caption)
		print(caption)
		plt.show()

if __name__ == '__main__':
	train_dataset = Dataset("train")
	# 2471297228_b784ff61a2.jpg  3101796900_59c15e0edc.jpg  351876121_c7c0221928.jpg   97406261_5eea044056.jpg
	image_id = '351876121_c7c0221928'
	caption_generator = CaptionGenerator('model_0.h5', train_dataset.vocabulary)
	caption_generator.display_image_caption(
		image_id,
		train_dataset.features[image_id]
	)
