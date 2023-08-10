import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_embedding_heatmap(embeddings):
    embedding_dim = len(embeddings[0])
    num_embeddings = len(embeddings)

    # Plot heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(embeddings)

    # Set axis labels
    ax.set_xticks(np.arange(embedding_dim))
    ax.set_yticks(np.arange(num_embeddings))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(embeddings)):
        for j in range(len(embeddings[0])):
            text = ax.text(j, i, embeddings[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Embedding Matrix Heatmap")
    fig.tight_layout()
    plt.show()