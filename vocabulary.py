from keras.preprocessing.text import Tokenizer

class Vocabulary():
    def __init__(self, captions):
        self.captions = captions
        self.words = self._words()
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.words)
        self.word_to_index = self.tokenizer.word_index
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.caption_max_length = self._caption_max()

    def _words(self):
        all_words = []
        for id, caption_set in self.captions.items():
            all_words.extend(" ".join(caption_set).split(" "))
        return list(set(all_words))

    def _caption_max(self):
        max_len = 0
        for caption_set in self.captions.values():
            _max = max([len(caption.split(" ")) for caption in caption_set])
            if _max > max_len:
                max_len = _max
        return max_len
