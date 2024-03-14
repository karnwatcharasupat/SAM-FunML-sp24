import os.path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from skimage import measure

def rough_check():

    bad_images = []
    ok_images = []

    samples = np.load("data/samples.npy", allow_pickle=True)
    labels = np.load("data/labels.npy", allow_pickle=True)

    if os.path.exists("data/bad_images.npy"):
        bad_images = list(np.load("data/bad_images.npy"))
    else:
        bad_images = []

    if os.path.exists("data/maybe_images.npy"):
        ok_images = list(np.load("data/maybe_images.npy"))
    else:
        ok_images = []

    if 150 in ok_images:
        ok_images.remove(150)

    original_bad_images = bad_images.copy()
    original_ok_images = ok_images.copy()

    np.save("data/original_bad_images.npy", np.array(original_bad_images))
    np.save("data/original_maybe_images.npy", np.array(original_ok_images))

    n_images = labels.shape[0]

    pos_area = np.zeros((n_images,))

    for i in tqdm(range(n_images)):

        if i in bad_images:
            continue

        if i in ok_images:
            continue

        label = labels[i].astype(float)
        total_area = label.shape[0] * label.shape[1]

        blobs = measure.label(label, background=0)
        n_blobs = np.max(blobs)

        if n_blobs > 1:
            print(f"Image {i} has {n_blobs} blobs")

        blob_ratio = np.zeros((n_blobs,))

        for b in range(n_blobs):
            blob = blobs == b + 1
            blob_area = np.sum(blob)
            blob_ratio[b] = 100 * blob_area / total_area

        threshold = 0.05
        small_blobs = blob_ratio < threshold
        if np.any(blob_ratio <  threshold):
            print(f"Image {i} has small blobs")

            f, ax = plt.subplots(1, 3)

            ax[0].imshow(label)
            ax[1].imshow(blobs)
            for b in range(n_blobs):
                if small_blobs[b]:
                    blob = blobs == b + 1
                    ax[1].plot(np.where(blob)[1], np.where(blob)[0], "rx")

            ax[2].imshow(samples[i])

            plt.show(block=False)

            is_fine = input("Is it fine? (y/n): ")
            if is_fine == "n":
                bad_images.append(i)
            else:
                ok_images.append(i)
            plt.close("all")

    np.save("data/bad_images.npy", np.array(bad_images))
    np.save("data/maybe_images.npy", np.array(ok_images))

def double_check():

    bad_images = list(np.load("data/bad_images.npy"))
    ok_images = list(np.load("data/maybe_images.npy"))

    samples = np.load("data/samples.npy", allow_pickle=True)
    labels = np.load("data/labels.npy", allow_pickle=True)

    for i in bad_images:
        label = labels[i].astype(float)

        f, ax = plt.subplots(1, 2)

        ax[0].imshow(samples[i])
        ax[1].imshow(label)

        plt.show(block=False)

        is_fine = input("Is it fine? (y/n): ")
        if is_fine == "y":
            ok_images.append(i)
            bad_images.remove(i)
        plt.close("all")

    for i in ok_images:
        label = labels[i].astype(float)

        f, ax = plt.subplots(1, 2)

        ax[0].imshow(samples[i])
        ax[1].imshow(label)

        plt.show(block=False)

        is_fine = input("Is it fine? (y/n): ")
        if is_fine == "n":
            bad_images.append(i)
            ok_images.remove(i)
        plt.close("all")

    np.save("data/bad_images.npy", np.array(bad_images))
    np.save("data/maybe_images.npy", np.array(ok_images))

def check_bad():

    bad_images = list(np.load("data/bad_images.npy"))

    print(f"Bad images: {bad_images}")

    print(f"Number of bad images: {len(bad_images)}")

if __name__ == "__main__":
    import fire

    fire.Fire()