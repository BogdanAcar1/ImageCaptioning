from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import flickr8k_helper as fh
from numpy import argmax

model = load_model("model_0.h5")

def max_caption_length(captions):
    max_len = 0
    for caption_set in captions.values():
        _max = max([len(caption.split(" ")) for caption in caption_set])
        if _max > max_len:
            max_len = _max
    return max_len

def generate_caption(model, features, vocab, max_len):
    caption = "<start>"
    w2i = fh.word_to_index(vocab)
    i2w = fh.index_to_word(vocab)
    for i in range(max_len):
        idx_caption = [w2i[word] for word in caption.split(" ")]
        idx_caption = pad_sequences([idx_caption], max_len)
        target_idx = argmax(model.predict([features, idx_caption], verbose = 0))
        word = i2w[target_idx]
        caption += f" {word}"
        if word == "<end>":
            break
    return caption

if __name__ == '__main__':
    print("Loading captions...")
    train_captions, test_captions, validation_captions = fh.load_captions(split = True)
    print("Loading image features...")
    train_features, test_features, validation_features = fh.load_features(split = True)

    all_captions = {**train_captions, **test_captions, **validation_captions}
    max_len = max_caption_length(all_captions)
    vocab = fh.vocabulary(all_captions)

    ids = 0
    for id in test_captions.keys():
        predicted = generate_caption(model, test_features[id], vocab, max_len)
        print(f"predicted: {predicted}")#, actuals = {test_captions[id]}")
        ids += 1
        if ids > 3:
            break
