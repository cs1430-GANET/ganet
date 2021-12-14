import tensorflow as tf
import numpy as np
from skimage.io import imread_collection

from cgModel import CycleGan
from generator import Generator
from discriminator import Discriminator
from plots import *


def main():
    monet_generator = Generator()
    photo_generator = Generator()

    monet_discriminator = Discriminator()
    photo_discriminator = Discriminator()

    # def discriminator_loss(real, generated):
    #     real_loss = tf.keras.losses.BinaryCrossentropy(
    #         from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

    #     generated_loss = tf.keras.losses.BinaryCrossentropy(
    #         from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

    #     total_disc_loss = real_loss + generated_loss

    #     return total_disc_loss * 0.5

    # def generator_loss(generated):
    #     return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

    # def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    #     loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    #     return LAMBDA * loss1

    # def identity_loss(real_image, same_image, LAMBDA):
    #     loss = tf.reduce_mean(tf.abs(real_image - same_image))

    #     return LAMBDA * 0.5 * loss

    # monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    # photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # monet_discriminator_optimizer = tf.keras.optimizers.Adam(
    #     2e-4, beta_1=0.5)
    # photo_discriminator_optimizer = tf.keras.optimizers.Adam(
    #     2e-4, beta_1=0.5)

    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator
    )

    # cycle_gan_model.compile(
    #     m_gen_optimizer=monet_generator_optimizer,
    #     p_gen_optimizer=photo_generator_optimizer,
    #     m_disc_optimizer=monet_discriminator_optimizer,
    #     p_disc_optimizer=photo_discriminator_optimizer,
    #     gen_loss_fn=generator_loss,
    #     disc_loss_fn=discriminator_loss,
    #     cycle_loss_fn=calc_cycle_loss,
    #     identity_loss_fn=identity_loss
    # )

    checkpoint_dir = '../checkpoints/'

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if checkpoint is not None:
        cycle_gan_model.load_weights(checkpoint)
    else:
        print("No checkpoint found")

    # path = r'../data/photo_jpg/0a0c3a6d07.jpg'
    col_dir = '../data/photo_jpg/*.jpg'
    col = imread_collection(col_dir)
    for img in col:
        # test_image = tf.image.decode_jpeg(
        #     open('../data/photo_jpg/0a0c3a6d07.jpg', 'rb').read(), channels=3)

        test_image = np.expand_dims(img, axis=0)
        print(test_image.shape)

        result = monet_generator(test_image)
        show_photo_and_monet(test_image, result)

        break


main()
