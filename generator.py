import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from dataset import Dataset
from model import create_lstm

class DataGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def _generate_item(self, features, captions):
        x_features, x_captions, y = [], [], []
        for caption in captions:
            sequence = self.dataset.vocabulary.tokenizer.texts_to_sequences([caption])[0]
            for index in range(1, len(sequence)):
                prefix = pad_sequences(
                    [sequence[:index]],
                    maxlen = self.dataset.vocabulary.caption_max_length
                )[0]
                target = to_categorical(
                    [sequence[index]],
                    num_classes = len(self.dataset.vocabulary.tokenizer.word_index) +1
                )[0]
                x_features.append(features)
                x_captions.append(prefix)
                y.append(target)
        return np.array(x_features), np.array(x_captions), np.array(y)

    def generate(self):
        while True:
            for id in self.dataset.captions.keys():
                features = self.dataset.features[id][0]
                captions = self.dataset.captions[id]
                x_features, x_captions, y = self._generate_item(
                    features = features,
                    captions = captions,
                )
                yield [[x_features, x_captions], y]

def train_generator(generator, epochs = 1):
    """
    TODO: use model.create_lstm and add dynamic shape params
    """
    features_shape = list(train_dataset.features.values())[0][0].shape
    captions_shape = (generator.dataset.vocabulary.caption_max_length, )
    output_size = len(generator.dataset.vocabulary.tokenizer.word_index) + 1
    steps_per_epoch = len(generator.dataset.captions)
    lstm = create_lstm(
        features_shape,
        captions_shape,
        output_size
    )
    yielder = generator.generate()
    for i in range(epochs):
        lstm.fit_generator(
            yielder,
            epochs = epochs,
            steps_per_epoch = steps_per_epoch,
            verbose = 1
        )
        model.save('model_' + str(i) + '.h5')

if __name__ == '__main__':
    train_dataset = Dataset("train")
    print(list(train_dataset.features.values())[0][0].shape)
    generator = DataGenerator(train_dataset)
    train_generator(generator)
