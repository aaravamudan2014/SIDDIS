
from matplotlib import pyplot as plt
from Downscaler import *

def main():
    visualize = False

    test_image = [np.random.random((8, 8)) for i in range(10)]
    topo_image_1 = [np.random.random((64, 64)) for i in range(10)]
    topo_image_2 = [np.random.random((64, 64)) for i in range(10)]

    for j in range(len(test_image)):
        poor_mans_downscaled_image = generate_downscaled_image(test_image[j], topo_image_1[j], topo_image_2[j],
                                                               distance_ranking, "L1")
        if visualize:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(test_image[j])
            axs[0, 1].imshow(topo_image_2[j])
            axs[1, 0].imshow(topo_image_1[j])
            axs[1, 1].imshow(poor_mans_downscaled_image)
            plt.show()


if __name__ == '__main__':
    main()


