import tensorflow as tf

from preprocess import *
from plots import *
from generator import Generator
from discriminator import Discriminator
from cgModel import CycleGan


def main():
    strategy, AUTOTUNE = connect_tpu()

    MONET_FILENAMES, PHOTO_FILENAMES = get_data()

    # Get a batch of images
    monet_ds = load_dataset(MONET_FILENAMES, AUTOTUNE, labeled=True).batch(1)
    photo_ds = load_dataset(PHOTO_FILENAMES, AUTOTUNE, labeled=True).batch(1)

    example_monet = next(iter(monet_ds))
    example_photo = next(iter(photo_ds))

    # a sample phot and monet image
    show_photo_and_monet(example_photo, example_monet)

    with strategy.scope():
        monet_generator = Generator()  # transforms photos to Monet-esque paintings
        photo_generator = Generator()  # transforms Monet paintings to be more like photos

        # differentiates real Monet paintings and generated Monet paintings
        monet_discriminator = Discriminator()
        # differentiates real photos and generated photos
        photo_discriminator = Discriminator()

    # untrained plot
    to_monet = monet_generator(example_photo)
    show_photo_and_monet(example_photo, to_monet)

    with strategy.scope():
        # losses calculation
        def discriminator_loss(real, generated):
            real_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

            generated_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

            total_disc_loss = real_loss + generated_loss

            return total_disc_loss * 0.5

        def generator_loss(generated):
            return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

        def calc_cycle_loss(real_image, cycled_image, LAMBDA):
            loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

            return LAMBDA * loss1

        def identity_loss(real_image, same_image, LAMBDA):
            loss = tf.reduce_mean(tf.abs(real_image - same_image))

            return LAMBDA * 0.5 * loss

        # training
        monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        monet_discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        photo_discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)

        cycle_gan_model = CycleGan(
            monet_generator, photo_generator, monet_discriminator, photo_discriminator
        )

        cycle_gan_model.compile(
            m_gen_optimizer=monet_generator_optimizer,
            p_gen_optimizer=photo_generator_optimizer,
            m_disc_optimizer=monet_discriminator_optimizer,
            p_disc_optimizer=photo_discriminator_optimizer,
            gen_loss_fn=generator_loss,
            disc_loss_fn=discriminator_loss,
            cycle_loss_fn=calc_cycle_loss,
            identity_loss_fn=identity_loss
        )

    cycle_gan_model.fit(
        tf.data.Dataset.zip((monet_ds, photo_ds)),
        epochs=25
    )

main()
