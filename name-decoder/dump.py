#%%
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    

print(word_ids[:i+1])
        word_ids_tensor = torch.tensor(word_ids[:i+1])
        print(word_ids_tensor)
        word_embeddings = embedding(word_ids_tensor)
        print(word_embeddings)
        word_attention = attention(word_embeddings, word_embeddings, word_embeddings, mask=None)
        print(word_attention)
        plot_embedding_heatmap(word_attention.detach().numpy())
        plt.show()