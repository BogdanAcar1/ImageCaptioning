from keras.layers import Input, Dropout, Dense, Embedding, LSTM, Add
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

IMG_FEATURES_SHAPE = (512, )

def create_lstm(input_size_1, input_size_2, output_size):
    print(input_size_1, input_size_2, output_size)
    input_layer1 = Input(shape = input_size_1)
    input_layer2 = Input(shape = input_size_2)
    feature_extraction_layer1 = Dropout(0.5)(input_layer1)
    feature_extraction_layer2 = Dense(256, activation = "relu")(feature_extraction_layer1)
    sequence_layer1 = Embedding(output_size, 256, mask_zero = True)(input_layer2)
    sequence_layer2 = Dropout(0.5)(sequence_layer1)
    sequence_layer3 = LSTM(256)(sequence_layer2)
    decoder_layer1 = Add()([feature_extraction_layer2, sequence_layer3])
    decoder_layer2 = Dense(256, activation = "relu")(decoder_layer1)
    output_layer = Dense(output_size, activation = "softmax")(decoder_layer2)
    lstm = Model(inputs = [input_layer1, input_layer2], outputs = output_layer)
    lstm.compile(loss = "categorical_crossentropy", optimizer = "adam")
    print(lstm.summary())
    plot_model(lstm, to_file = "lstm.png", show_shapes = True)
    return lstm

if __name__ == '__main__':
    lstm = create_lstm(8000, 34)
