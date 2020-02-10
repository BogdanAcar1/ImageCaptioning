from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input

vgg16_encoder = VGG16(
    include_top = False,
    input_tensor = Input(shape = (224, 224, 3)),
    pooling = "avg"
)

def process_image(image):
    """
    Map image to VGG16 input shape.
    """
    image = img_to_array(load_img(image, target_size = (224, 224)))
    return preprocess_input(
        image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    )

def encode(image):
    return vgg16_encoder.predict(process_image(image), verbose = 0)

if __name__ == '__main__':
    model = vgg16_encoder()
