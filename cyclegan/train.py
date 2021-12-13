import tensorflow as tf

from preprocess import *
from plots import *


def main():
    strategy, AUTOTUNE = connect_tpu()

    MONET_FILENAMES, PHOTO_FILENAMES = get_data()

    # Get a batch of images
    monet_ds = load_dataset(MONET_FILENAMES, AUTOTUNE, labeled=True).batch(1)
    photo_ds = load_dataset(PHOTO_FILENAMES, AUTOTUNE, labeled=True).batch(1)

    example_monet = next(iter(monet_ds))
    example_photo = next(iter(photo_ds))

    show_photo_and_monet(example_photo, example_monet)


main()
