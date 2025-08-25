# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn

from fairseq import utils


class OPRPositionalEmbedding(nn.Module):
    """This module produces Orthogonal Position Representation (OPR) with controllable coefficient k.
    Padding symbols are ignored, with support for left/right padding specification.
    
    Core OPR Formula (d: embedding dimension, pos: position index, j: dimension index, k: control coefficient):
    P(pos, 2j) = sin(pos · (2π/k) · (2j/d))
    P(pos, 2j+1) = cos(pos · (2π/k) · (2j/d))
    where j ∈ [0, d/2), k > 1 (controls orthogonality range: larger k → narrower similar position range)
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, k=8, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim  # Corresponding to `d` in OPR formula
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.k = k  # Control coefficient for orthogonality range (must be > 1, as per method definition)

        # Initialize OPR weights with the specified k
        self.weights = OPRPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
            self.k,  # Pass k to embedding generation
        )
        self.register_buffer('_float_tensor', torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None, k=2):
        """Build OPR embeddings with controllable coefficient k, based on the proposed formula."""
        half_dim = embedding_dim // 2  # j ranges from 0 to half_dim-1 (matches [0, d/2))
        d = embedding_dim  # Alias for clarity (matches OPR formula notation)
        
        # Step 1: Compute OPR frequency term with coefficient k: (2π/k)·(2j/d) = 4πj/(k·d)
        # j is indexed by torch.arange(half_dim), so 2j = 2 × j_idx
        freq = torch.arange(half_dim, dtype=torch.float) * (2 * 2 * math.pi) / (k * d)  # Core term: (2π/k)·(2j/d)
        
        # Step 2: Generate position-wise embedding (pos × freq)
        # pos ranges from 0 to num_embeddings-1 (all possible positions)
        pos = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)  # [num_embeddings, 1]
        emb = pos * freq.unsqueeze(0)  # [num_embeddings, half_dim]
        
        # Step 3: Concatenate sin (even dims) and cos (odd dims) → strictly match OPR formula
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [num_embeddings, 2×half_dim]
        
        # Step 4: Handle odd embedding dimension (zero-pad last dim if needed, to match embedding_dim)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1, dtype=torch.float)], dim=1)
        
        # Step 5: Set padding position embedding to zero (consistent with original padding logic)
        if padding_idx is not None and padding_idx < num_embeddings:
            emb[padding_idx, :] = 0
        
        return emb

    def forward(self, input, incremental_state=None):
        """Input shape: [batch_size × sequence_length]. Output shape: [batch_size × sequence_length × embedding_dim]."""
        bsz, seq_len = input.size()
        # Calculate maximum required position (skip padding: padding_idx + 1 + seq_len covers all valid positions)
        max_pos = self.padding_idx + 1 + seq_len
        
        # Expand OPR weights if current max_pos exceeds pre-initialized size (reuse k from class init)
        if max_pos > self.weights.size(0):
            self.weights = OPRPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
                self.k,  # Ensure expanded weights use the same k
            ).type_as(self.weights)
        # Ensure weight dtype matches the model's float tensor type (avoid precision mismatch)
        self.weights = self.weights.type_as(self._float_tensor)

        # Incremental state: handle decoding single step (fixed position for all tokens in current step)
        if incremental_state is not None:
            target_pos = self.padding_idx + seq_len  # Current decoding position (skip padding)
            return self.weights[target_pos, :].expand(bsz, 1, -1)  # [bsz, 1, embedding_dim]

        # Generate valid position indices (skip padding positions via make_positions)
        positions = utils.make_positions(input.data, self.padding_idx, self.left_pad)  # [bsz, seq_len]
        # Retrieve OPR embeddings for valid positions and reshape to match input batch structure
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1)

    def max_positions(self):
        """Maximum supported position (arbitrarily large to fit long sequences, consistent with original design)."""
        return int(1e5)

    def update_k(self, new_k):
        """Optional method to update k dynamically (if needed for adaptive scenarios)."""
        self.k = new_k
        # Re-initialize weights with new k to apply update (reset to init_size for consistency)
        self.weights = OPRPositionalEmbedding.get_embedding(
            init_size=1024,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            k=self.k,
        ).type_as(self.weights)
