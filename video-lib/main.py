import cv2
import os

os.sys.path.insert(0, "../cyclegan/")

from cgModel import CycleGan
from generator import Generator
from discriminator import Discriminator
from multiprocessing import Process
import tensorflow as tf
import numpy as np

def return_generator():
    monet_generator = Generator()
    photo_generator = Generator()

    monet_discriminator = Discriminator()
    photo_discriminator = Discriminator()

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

    checkpoint_dir = '../checkpoints/'
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    cycle_gan_model.load_weights(checkpoint).expect_partial()

    return monet_generator

def monet_frame(image, generator, output=""):
    image = np.expand_dims(image, axis=0)
    result = np.array(generator(image))
    cv2.namedWindow("Monet", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Monet", result[0])
    cv2.resizeWindow("Monet", 512, 512)
    return

def main():
    if len(os.sys.argv) < 2 or os.sys.argv[1] == "":
        print("usage: python webcam/video")
        if len(os.sys.argv) == 3 and os.sys.argv[2] == "":
            print("usage: python video path-to-image-frames-directory")
        exit()

    def convert_cam_frames_to_monet(generator):
        while True:
            video = cv2.VideoCapture(0)
            _ret_val, img = video.read()
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (256, 256))

            normalization_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
            frame_nor = normalization_layer(img)
            monet_frame(frame_nor, generator)
            cv2.imshow('my webcam', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def convert_video_frames_to_monet(generator, path):
        cap = cv2.VideoCapture(path)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        count = 0
        # only monet every 60 frames
        while(cap.isOpened()):
            if count % 60 == 0:
                print(count)
                ret_val, img = cap.read()
                if ret_val:
                    img = cv2.resize(img, (256, 256))

                    normalization_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
                    frame_nor = normalization_layer(img)
                    monet_frame(frame_nor, generator)
                    cv2.namedWindow("Video", cv2.WINDOW_KEEPRATIO)
                    cv2.imshow("Video", img)
                    cv2.resizeWindow("Video", 512, 512)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
        cv2.destroyAllWindows()
        cap.release()

    if os.sys.argv[1] == "webcam":
        generator = return_generator()
        convert_cam_frames_to_monet(generator)
    
    if os.sys.argv[1] == "video":
        generator = return_generator()
        convert_video_frames_to_monet(generator, os.sys.argv[2])

    
if __name__ == "__main__":
    main()