#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor, BoolTensor
from torch.nn import functional as F
import time
from typing import Union, Optional  # <--- 添加 Optional
import os                            # <--- 添加 import os

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore
import json


class MCMark_WatermarkCode(AbstractWatermarkCode):
    # [FIX] 添加一个类变量作为缓存
    _cached_shuffle = None
    _cached_path = None

    def __init__(self, shuffle: LongTensor, split_k: BoolTensor):
        self.shuffle = shuffle
        self.split_k = split_k
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: Union[torch.Generator, list[torch.Generator]],
        vocab_size: int,
        split_num: int,
        shuffle_file_path: Optional[str] = None
    ):
        # 1. 尝试设置默认路径
        if shuffle_file_path is None and os.path.exists("semantic_shuffle_opt.pt"):
            shuffle_file_path = "semantic_shuffle_opt.pt"

        # 2. 检查缓存：如果路径没变且缓存已有，直接使用缓存
        loaded_shuffle = None
        if shuffle_file_path:
            if cls._cached_shuffle is not None and cls._cached_path == shuffle_file_path:
                # 命中缓存！直接使用，不打印日志，不读硬盘
                loaded_shuffle = cls._cached_shuffle
            elif os.path.exists(shuffle_file_path):
                # 未命中缓存，从硬盘加载
                try:
                    print(f"[MCMark] Loading semantic shuffle from {shuffle_file_path}...") # 这句话只会出现一次了
                    loaded_shuffle = torch.load(shuffle_file_path, map_location="cpu")
                    if len(loaded_shuffle) != vocab_size:
                        print(f"[MCMark] Warning: Size mismatch. Random fallback.")
                        loaded_shuffle = None
                    else:
                        # 存入缓存
                        cls._cached_shuffle = loaded_shuffle
                        cls._cached_path = shuffle_file_path
                except Exception as e:
                    print(f"[MCMark] Error loading file: {e}")
                    loaded_shuffle = None

        # 3. 构造 Shuffle Tensor (这部分保持原样)
        if isinstance(rng, list):
            batch_size = len(rng)
            if loaded_shuffle is not None:
                # 使用缓存的 shuffle，扩展到 batch 维度
                shuffle = loaded_shuffle.to(rng[0].device).unsqueeze(0).expand(batch_size, -1)
            else:
                shuffle = torch.stack(
                    [
                        torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                        for i in range(batch_size)
                    ]
                )
            
            split_k = torch.cat(
                [
                    torch.randint(
                        low=0,
                        high=split_num,
                        size=(1,),
                        dtype=torch.long,
                        generator=rng[i],
                        device=rng[i].device,
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            if loaded_shuffle is not None:
                shuffle = loaded_shuffle.to(rng.device)
            else:
                shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
                
            split_k = torch.randint(
                low=0,
                high=split_num,
                size=(1,),
                dtype=torch.long,
                device=rng.device,
                generator=rng,
            )
            
        return cls(shuffle, split_k)

    # @classmethod
    # def from_random(
    #     cls,
    #     rng: Union[torch.Generator, list[torch.Generator]],
    #     vocab_size: int,
    #     split_num: int,
    # ):
    #     if isinstance(rng, list):
    #         batch_size = len(rng)
    #         shuffle = torch.stack(
    #             [
    #                 torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
    #                 for i in range(batch_size)
    #             ]
    #         )
    #         split_k = torch.cat(
    #             [
    #                 torch.randint(
    #                     low=0,
    #                     high=split_num,
    #                     size=(1,),
    #                     dtype=torch.long,
    #                     generator=rng[i],
    #                     device=rng[i].device,
    #                 )
    #                 for i in range(batch_size)
    #             ],
    #             dim=0,
    #         )
    #     else:
    #         shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
    #         split_k = torch.randint(
    #             low=0,
    #             high=split_num,
    #             size=(1,),
    #             dtype=torch.long,
    #             device=rng.device,
    #             generator=rng,
    #         )
    #     return cls(shuffle, split_k)

        


class MC_Reweight(AbstractReweight):
    watermark_code_type = MCMark_WatermarkCode

    def __init__(self, n: float):
        self.n = n

    def __repr__(self):

        return f"MC_Reweight(n={self.n})"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        """
        \textbf{$\gamma$-reweight:}
        Let the watermark code space $\mathcal{E}$ be the set of all bijective function between symbol set $\Sigma$ and a set of number $[\abs{\Sigma}]=\{1,\dots,\abs{\Sigma}\}$, where $\abs{\Sigma}$ is the size of symbol set $\Sigma$.
        Essentially, any watermark code $E$ is an indexing function for symbol set $\Sigma$, and also assign an order on $\Sigma$. Let $P_E$ be the uniform probability on $\mathcal{E}$, it would be easy to sample a watermark code $E$ by randomly shuffle the symbol list.

        Assume the original distribution is $P_T(t)\in\Delta_\Sigma,\forall t\in\Sigma$.
        We interpret watermark code $E:\Sigma\to[\abs{\Sigma}]$ as a indexing function and we introduce parameter $\gamma$ to control the strength of watermark.
        % Use the hash of
        % $E$ as a pseudorandom number seed and sample a random permutation $\sigma:\Sigma\to N$.
        Then we construct auxiliary functions
        % $F_I(i)=P_{t\sim P_T}(E(t)\leq i),$
        $F_I(i)=\sum_{t\in\Sigma} \mathbf{1}(E(t)\leq i) P_T(t),$
        $F_S(s)=\begin{cases}(1-\gamma)s & s\leq\frac{1}{2}\\-\gamma+(1+\gamma)s ~~~& s>\frac{1}{2}\end{cases},$
        $F_{I'}(i)=F_S(F_I(i)).$
        The new distribution is given by $P_{T'}(t)=F_{I'}(E(t))-F_{I'}(E(t)-1)$.
        """

        def set_nan_to_zero(x):
            x[torch.isnan(x)] = 0
            return x

        start = time.time()
        # s_ means shuffled
        s_logits = torch.gather(p_logits, -1, code.shuffle)
        s_probs = torch.softmax(s_logits, dim=-1)
        bsz, vocab_size = s_logits.shape

        splits = []
        if self.n == vocab_size:
            splits = [[i] for i in range(self.n)]
        elif vocab_size % self.n == 0:
            splits = (
                torch.arange(start=0, end=vocab_size)
                .reshape(self.n, vocab_size // self.n)
                .to(p_logits.device)
            )
        else:
            for n_idx in range(self.n):
                splits.append(
                    list(
                        range(
                            round(vocab_size * n_idx / self.n),
                            round(vocab_size * (n_idx + 1) / self.n),
                        )
                    )
                )

        split_k = code.split_k.to(s_logits.device)

        split_sums = []
        if self.n == vocab_size:
            split_sums = s_probs
        elif vocab_size % self.n == 0:
            split_sums = s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
        else:
            for n_idx in range(self.n):
                cur_split = splits[n_idx]
                split_sums.append(s_probs[:, cur_split].sum(dim=-1, keepdim=True))

            split_sums = torch.cat(split_sums, dim=-1)  # [bsz,n]
        scales = torch.minimum(
            self.n * torch.ones_like(split_sums).to(s_probs.device), 1 / split_sums
        )  # [bsz,n]

        overflow_scales = (
            self.n * split_sums - 1
        ) / split_sums  # [bsz,n] note: might be negative or nan
        overflow_scales = set_nan_to_zero(overflow_scales)
        overflow_scales[overflow_scales < 0] = 0  # [bsz,n]

        target_scales = scales[range(bsz), split_k]  # [bsz]
        target_sums = split_sums[range(bsz), split_k]  # [bsz]

        remain_sums = 1 - target_scales * target_sums  # [bsz]
        overflow_sums = (overflow_scales * split_sums).sum(dim=-1)  # [bsz]
        fill_scale = remain_sums / overflow_sums  # [bsz]
        fill_scale = set_nan_to_zero(fill_scale)  # [bsz]

        split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(
            bsz, 1
        ) == split_k.view(-1, 1).repeat(1, self.n)
        final_scale = torch.where(
            split_mask,
            target_scales.view(-1, 1).repeat(1, self.n),
            fill_scale.view(-1, 1) * overflow_scales,
        )  # [bsz,n]

        reweighted_s_probs = torch.zeros_like(s_probs).to(s_logits.device)

        if self.n == vocab_size:
            reweighted_s_probs = final_scale * s_probs
        elif vocab_size % self.n == 0:
            reweighted_s_probs = (
                final_scale.view(bsz, self.n, 1)
                .expand((-1, -1, vocab_size // self.n))
                .reshape(bsz, vocab_size)
                * s_probs
            )
        else:
            for n_idx in range(self.n):
                cur_split = splits[n_idx]
                reweighted_s_probs[:, cur_split] = (
                    final_scale[:, n_idx].view(-1, 1) * s_probs[:, cur_split]
                )

        reweighted_s_probs[reweighted_s_probs < 0] = 0

        reweighted_s_logits = torch.log(reweighted_s_probs)
        reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)

        return reweighted_logits
