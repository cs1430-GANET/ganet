# Abstract

Artistic work is generally recognized through an artist’s brush strokes, color choices, color palette and so on. Each artist’s style is unique and hard to replicate. With advancements in computer vision however, Generative Adversarial Networks are now able to mimic such nuances in a very convincing way. We intend to use GANs (specifically Cycle GAN) to replicate Claude Monet’s artistic style and see if our GAN can create realistic renditions of his work by transforming normal photos, as well as paintings by other famous artists.

## To Run Code

You can run the base code in two ways:

1. Run train.py in cyclegan and then test.py
2. Run the colab notebook cgan_base_code.ipynb

test.py has code that can pull in saved weights.

Other colab notebooks can be run too to experiment with different artists and datasets.

## Video output on trained model

To run the model weights to convert a normal video to monet style video:

```python
    cd video-lib
    python main.py video path-to-video
```

For webcam:

```python
    cd video-lib
    python main.py webcam
```

NOTE: works best on nature/scenery video frames

For reference, there's already a generated video in video-lib:

[Watch the video](https://drive.google.com/file/d/1hbPl5qvVE-T613iPyFf_LHhV1_DZuXzb/view?usp=sharing)
