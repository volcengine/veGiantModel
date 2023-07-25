import torch
import torch.nn as nn


class GPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx, device=device, dtype=dtype
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, word_embed_proj_dim, padding_idx=padding_idx, device=device, dtype=dtype
            )
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim, device=device, dtype=dtype)

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

    def reset_parameters(self):
        # TODO: check if this is necessary
        def _init_weights(module, initializer_range=0.02):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=initializer_range)

        self.apply(_init_weights)
