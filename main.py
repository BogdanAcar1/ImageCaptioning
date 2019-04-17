import flickr8k_helper as fh
from keras.utils import to_categorical
import lstm
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

def max_caption_length(captions):
    max_len = 0
    for caption_set in captions.values():
        _max = max([len(caption.split(" ")) for caption in caption_set])
        if _max > max_len:
            max_len = _max
    return max_len

# create batches - current approach causes MemoryError
def generate_data_item(features, captions, vocab, max_len):
    x_img = []
    x_capt = []
    y = []
    w2i = fh.word_to_index(vocab)
    for caption in captions:
        idx_caption = [w2i[word] for word in caption.split(" ")]
        for split in range(1, len(idx_caption)):
            prefix = idx_caption[:split]
            target = idx_caption[split]
            prefix = pad_sequences([prefix], maxlen = max_len)[0]
            target = to_categorical([target], num_classes = len(vocab) + 1)[0]
            x_img.append(features)
            x_capt.append(prefix)
            y.append(target)
    return np.array(x_img), np.array(x_capt), np.array(y)

def dataset_generator(features, captions, vocab, max_len):
    """
    Some off-the-cuff ideas for further improving this data generator include:
    Randomize the order of photos each epoch.
    Work with a list of photo ids and load text and photo data as needed to cut even further back on memory.
    Yield more than one photo’s worth of samples per batch.
    """
    while True:
        for id in captions.keys():
            img_features = features[id][0]
            img_captions = captions[id]
            x_img, x_capt, y = generate_data_item(img_features, img_captions, vocab, max_len)
            yield [[x_img, x_capt], y]

def train_generator(model, features, captions, vocab, max_len, epochs = 5):
    steps_per_epoch = len(captions)
    for i in range(epochs):
        generator = dataset_generator(features, captions, vocab, max_len)
        model.fit_generator(generator, epochs = 1, steps_per_epoch = len(captions), verbose = 1)
        model.save('model_' + str(i) + '.h5')

if __name__ == '__main__':

    print("Loading captions...")
    train_captions, test_captions, validation_captions = fh.load_captions(split = True)
    print("Loading image features...")
    train_features, test_features, validation_features = fh.load_features(split = True)

    all_captions = {**train_captions, **test_captions, **validation_captions}
    max_len = max_caption_length(all_captions)
    vocab = fh.vocabulary(all_captions)

    print("Starting training ...")
    model = lstm.create_lstm(len(vocab) + 1, max_len)
    train_generator(model, train_features, train_captions, vocab, max_len)

    # print("Creating validation input set...")
    # x_valid_img, x_valid_capt, y_valid = create_dataset(validation_features, validation_captions, vocab, max_len)
    # print("Training model...")
    # model = lstm.create_lstm(len(vocab), max_len)
    # filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    # checkpoint = ModelCheckpoint(filepath, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
    # model.fit([x_train_img, x_train_capt], y_train, epochs = 1, verbose = 2, callbacks = [checkpoint], validation_data = ([x_valid_img, x_valid_capt], y_valid))
