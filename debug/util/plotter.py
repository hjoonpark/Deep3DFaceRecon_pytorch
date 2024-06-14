import os
import matplotlib.pyplot as plt

def plot_recons(Is, save_path, suptitle, titles, n_rows):
    """
    Input
        Is: np.ndarray (B, H, W, C) [0, 255] uint8
        suptitle: str
        titles: list[str]
    """
    R = n_rows
    C = 1
    W = (Is.shape[2]+1) // 224
    L = 3
    fig = plt.figure(figsize=(L*C*W, L*R))

    for b in range(Is.shape[0]):
        if b >= R:
            break
        I = Is[b]
        ax = fig.add_subplot(R, C, b+1)
        ax.imshow(I)
        ax.set_title(titles[b])

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()




