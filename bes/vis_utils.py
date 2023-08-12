import matplotlib.pyplot as plt

def plot_attention_heatmap(attn):
    attn = attn.detach().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(attn)
    fig.tight_layout()
    plt.show()

def plot_output_heatmap(output, reference, candidate):
    output = output.detach().numpy()
    fig, ax = plt.subplots()
    ax.set_xticks(range(len(reference)))
    ax.set_xticklabels(reference)
    ax.set_yticks(range(len(candidate)))
    ax.set_yticklabels(candidate)
    im = ax.imshow(output)
    fig.tight_layout()
    plt.show()