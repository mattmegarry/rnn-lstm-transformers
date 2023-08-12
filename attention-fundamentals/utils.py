import matplotlib.pyplot as plt

def plot_attention_heatmap(attn):
    attn = attn.detach().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(attn)
    fig.tight_layout()
    plt.show()