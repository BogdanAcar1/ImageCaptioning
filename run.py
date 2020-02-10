import flickr8k_helper as fh

if __name__ == '__main__':
    captions = fh.split_captions("train")
    print(len(captions))
    features = fh.split_features("train")
    print(len(features))
