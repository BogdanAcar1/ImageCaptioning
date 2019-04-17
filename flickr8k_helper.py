import string
import os
import vgg16
import pickle
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer

FLICKR8K_IMAGES_PATH = "Flicker8k_Dataset"
FLICKR8K_CAPTIONS_PATH = "Flickr8k_text/Flickr8k.token.txt"
FLICKR8K_TRAIN = "Flickr8k_text/Flickr_8k.trainImages.txt"
FLICKR8K_VALIDATION = "Flickr8k_text/Flickr_8k.devImages.txt"
FLICKR8K_TEST = "Flickr8k_text/Flickr_8k.testImages.txt"

CAPTIONS_OUT_PATH = "captions.txt"
FEATURES_OUT_PATH = "features.pkl"
VOCABULARY_OUT_PATH = "vocabulary.txt"

SPLIT_SETS = ["train", "test", "validation"]

def normalize_caption(caption):
    """
    Normalize caption by lowercasing tokens,
    removing punctuation and single character tokens .
    """
    tokens = [token.lower() for token in caption.split() if len(token) > 1]
    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
    return " ".join(tokens)

def read_captions(caption_file = FLICKR8K_CAPTIONS_PATH, split = False):
    """
    Loads captions in caption_file and returns a dictionary where keys
    are image ids and values are the multiple normalized captions of the image
    with the specific id.
    """
    captions = {}
    captions_in = open(caption_file, "r")
    caption_lines = [(line.split("\t")[0].split(".")[0], normalize_caption(line.split("\t")[1])) for line in captions_in.readlines()]
    captions_in.close()
    for (id, caption) in caption_lines:
        caption = "<start> " + caption + " <end>"
        if id not in captions:
            captions[id] = [caption]
        else:
            captions[id].append(caption)
    if split == False:
        return captions
    else:
        captions = [subset(captions, split = split_set) for split_set in SPLIT_SETS]
        return captions[0], captions[1], captions[2]

def dump_captions(captions, caption_out_file = CAPTIONS_OUT_PATH):
    """
    Writes processed captions from original dataset to file to later use in language
    model training.
    """
    captions_out = open(caption_out_file, "w")
    for (id, caption_set) in captions.items():
        for caption in caption_set:
            captions_out.write(f"{id}\t{caption}\n")
    captions_out.close()

def load_captions(caption_in_file = CAPTIONS_OUT_PATH, split = False):
    captions_in = open(caption_in_file, "r")
    captions = {}
    for line in captions_in.readlines():
        id = line.split("\t")[0]
        if id in captions.keys():
            captions[id].append(line.split("\t")[1])
        else:
            captions[id] = [line.split("\t")[1]]
    captions_split = [[], [], []]
    captions_in.close()
    if split == True:
        subsets = split_into_subsets(captions)
        return subsets[0], subsets[1], subsets[2]
    return captions

def encode_images(images_path = FLICKR8K_IMAGES_PATH, max_img_num = 10000):
    """
    Loads images in images_path directory and returns a dictionary where keys are
    image ids and values are features of the specific image extracted using VGG16.
    """
    features = {}
    for img_num, image_in in tqdm(enumerate(os.listdir(images_path))):
        features[image_in.split(".")[0]] = vgg16.encode((os.path.join(images_path, image_in)))
        if img_num >= max_img_num:
            break
    return features

def dump_features(features, features_out_file = FEATURES_OUT_PATH, split = False):
    """
    Write serialized features to file in order to avoid recurrent feature extraction
    before each attempt of training the language model.
    If split == True, dumps the features into three separate file, each corresponding
    to a different split set: training, test, validation.
    Otherwise, dumps all features into a single file.
    """
    if split == False:
        features_out = open(features_out_file, "wb")
        pickle.dump(features, features_out)
        features_out.close()
    else:
        split_ids = {split_set: load_subset_ids(split = split_set) for split_set in SPLIT_SETS}
        for split_set in SPLIT_SETS:
            with open(f"features_{split_set}.pkl", "wb") as features_out:
                pickle.dump({id: feature for id, feature in features.items() if id in split_ids[split_set]}, features_out)

def load_features(features_in_file = FEATURES_OUT_PATH, split = False):
    """
    Load serialized features from file.
    If split == True, returns the three split sets: training, test and validation.
    Otherwise, returns the entire feature set
    """
    if split == False:
        features_in = open(features_in_file, "rb")
        features = pickle.load(features_in)
        features_in.close()
        return features
    else:
        features = []
        for split_set in SPLIT_SETS:
            with open(f"features_{split_set}.pkl", "rb") as features_in:
                features.append(pickle.load(features_in))
        return features[0], features[1], features[2]

def vocabulary(captions = None):
    """
    Computes the set of distinct words over all captions.
    """
    if captions == None:
        captions = read_captions()
    all_words = []
    for id, caption_set in captions.items():
        all_words.extend(" ".join(caption_set).split(" "))
    return list(set(all_words))

def word_to_index(vocab = None):
    if vocabulary == None:
        vocab = vocabulary()
    return {word: id + 1 for id, word in enumerate(vocab)}

def index_to_word(vocab = None):
    if vocabulary == None:
        vocab = vocabulary()
    return {id + 1: word for id, word in enumerate(vocab)}

def load_subset_ids(split = "train", max_num = 100000):
    if split == "train":
        subset_file = FLICKR8K_TRAIN
    elif split == "test":
        subset_file = FLICKR8K_TEST
    else:
        subset_file = FLICKR8K_VALIDATION
    subset_in = open(subset_file, "r")
    ids = [line.split(".")[0] for line in subset_in.readlines()]
    subset_in.close()
    if len(ids) > max_num:
        ids = ids[:max_num]
    return ids

def subset(data, split = "train", max_num = 100000):
    return {id: value for id, value in data.items() if id in load_subset_ids(split, max_num = max_num)}

def split_into_subsets(data):
    subsets = []
    subset_ids = []
    for split in SPLIT_SETS:
        subsets.append({})
        subset_ids.append(load_subset_ids(split = split))
    for id, val in data.items():
        for i, split in enumerate(SPLIT_SETS):
            if id in subset_ids[i]:
                subsets[i][id] = val
    return subsets

if __name__ == '__main__':
    #train, test, val = read_captions(split = True)
    #print([len(s) for s in [train, test, val]])
    x, y, z = load_captions(split = True)
    print(len(x), len(y), len(z))
