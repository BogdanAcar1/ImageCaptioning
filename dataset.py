import pickle
import string
from collections import defaultdict
from vocabulary import Vocabulary

FLICKR8K_IMAGES_PATH = "Flicker8k_Dataset"
FLICKR8K_CAPTIONS_PATH = "Flickr8k_text/Flickr8k.token.txt"
FLICKR8K_TRAIN = "Flickr8k_text/Flickr_8k.trainImages.txt"
FLICKR8K_VALIDATION = "Flickr8k_text/Flickr_8k.devImages.txt"
FLICKR8K_TEST = "Flickr8k_text/Flickr_8k.testImages.txt"

CAPTIONS_OUT_PATH = "captions.txt"
FEATURES_OUT_PATH = "features.pkl"
VOCABULARY_OUT_PATH = "vocabulary.txt"

class Dataset():
    def __init__(self, split = "all"):
        self.captions = self._split_subset(self._load_captions(), split)
        self.features = self._split_subset(self._load_features(), split)
        self.vocabulary = Vocabulary(self.captions)

    def _split_subset(self, full_set, split):
        """
        Split dataset into either training, testing, validation or all
        based on flickr8k partitioning
        """
        if split not in ["train", "test", "validation", "all"]:
            raise Exception(f"Unknown split parameter value '{split}'.")

        if split == "train":
            id_file = FLICKR8K_TRAIN
        elif split == "test":
            id_file = FLICKR8K_TEST
        elif split == "validation":
            id_file = FLICKR8K_VALIDATION
        elif split == all:
            return full_set
        else:
             return None
        with open(id_file, "r") as id_in:
            ids = [line.split(".")[0] for line in id_in.readlines()]
            return {id: full_set[id] for id in ids}

    def _load_captions(self, captions_file = CAPTIONS_OUT_PATH):
        """
        Load captions from file in a dictionary where keys are image ids
        and values their corresponding caption list.
        """
        captions_dict = defaultdict(list)
        with open(captions_file, "r") as captions_in:
            lines = [
                (line.split("\t")[0].split(".")[0], self._normalize_caption(line.split("\t")[1]))
                for line
                in captions_in.readlines()
            ]
        for id, caption in lines:
            captions_dict[id].append(caption)
        return captions_dict

    def _load_features(self, features_file = FEATURES_OUT_PATH):
        """
        Load serialized features from file.
        """
        with open(features_file, "rb") as features_in:
            return pickle.load(features_in)

    def _normalize_caption(self, caption):
        """
        Normalize caption by lowercasing tokens,
        removing punctuation and single character tokens .
        """
        tokens = [token.lower() for token in caption.split() if len(token) > 1]
        tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
        return " ".join(tokens)

if __name__ == '__main__':
    train_dataset = Dataset("train")
    print(len(train_dataset.captions))
    print(len(train_dataset.features))
