import matplotlib.pyplot as plt


def show_photo_and_monet(photo, monet):
    plt.subplot(121)
    plt.title('Photo')
    plt.imshow(photo[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Monet')
    plt.imshow(monet[0] * 0.5 + 0.5)

    plt.show()
