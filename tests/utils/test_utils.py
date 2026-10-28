# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from audiocraft.utils.utils import sample_top_k


def test_sample_top_k_returns_only_top_k_indices():
    probs = torch.tensor([[0.05, 0.15, 0.3, 0.5]])

    sampled = sample_top_k(probs, k=2)

    assert sampled.shape == (1, 1)
    assert sampled.item() in {2, 3}


def test_sample_top_k_does_not_modify_probabilities():
    probs = torch.tensor([[0.05, 0.15, 0.3, 0.5]])
    original_probs = probs.clone()

    sample_top_k(probs, k=2)

    assert torch.equal(probs, original_probs)


def test_sample_top_k_with_one_candidate_returns_maximum_index():
    probs = torch.tensor([[0.05, 0.15, 0.3, 0.5]])

    sampled = sample_top_k(probs, k=1)

    assert sampled.item() == 3
