# tools/generate_semantic_shuffle.py

import torch
import argparse
import os
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans

def generate_semantic_shuffle(model_name, split_num, save_path, softness=0.2, high_freq_ratio=0.2):
    print(f"Loading model: {model_name}...")
    # 加载模型以获取 Embedding
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    vocab_size = embeddings.shape[0]
    print(f"Vocab Size: {vocab_size}")

    # --- 1. 频率分层 (High Frequency Layer) ---
    # 在没有真实语料统计的情况下，我们假设 Tokenizer ID 较小或是 Embedding Norm 较低的往往是高频/功能词
    # 更严谨的做法是加载一个数据集统计词频，这里为了通用性，我们采用“前N个Token + 随机采样”作为近似
    # 或者直接简单粗暴：将整个词表视为需要处理的对象，但专门把 Top N ID 视为高频
    
    # 策略：前 20% 的 Token ID (通常包含常用词和特殊符号) 强制随机分配，不参与聚类
    # 注意：很多 Tokenizer (如 Llama) ID 不一定按频率排序。
    # 为了稳健，我们这里使用 Embedding 的 L2 Norm 作为一个简单的启发式特征（常用词往往 Norm 较稳定），
    # 但最稳健的“高频随机性”其实就是：随机选取 20% 的词作为“高频组”。
    
    indices = np.arange(vocab_size)
    np.random.shuffle(indices) # 先全打乱
    
    cutoff = int(vocab_size * high_freq_ratio)
    high_freq_indices = indices[:cutoff] # 这部分将保持完全随机 (High Entropy)
    semantic_indices = indices[cutoff:]  # 这部分将进行聚类 (Semantic Preservation)
    
    print(f"Processing: {len(high_freq_indices)} random tokens, {len(semantic_indices)} semantic tokens.")

    # --- 2. 语义聚类 (Mid/Low Frequency Layer) ---
    print("Clustering semantic tokens...")
    semantic_embeddings = embeddings[semantic_indices]
    
    # 我们需要的簇数量应该比通道数多，以便打散。比如每个通道包含 10 个簇。
    n_clusters = split_num * 20 
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init="auto")
    cluster_labels = kmeans.fit_predict(semantic_embeddings)

    # --- 3. 通道分配 (Channel Allocation with Softness) ---
    # 目标：构建一个 shuffle 数组。
    # 逻辑：MCMark 将 shuffle 后的数组切分为 split_num 份。
    # 所以我们需要把属于 Channel 0 的词放在数组最前面，Channel 1 的紧随其后...
    
    channel_buckets = [[] for _ in range(split_num)]
    
    # A. 分配高频词 (完全随机均匀分配)
    for idx in high_freq_indices:
        channel = np.random.randint(0, split_num)
        channel_buckets[channel].append(idx)
        
    # B. 分配语义词 (基于簇，带软划分)
    # 先给每个簇随机分配一个“主通道”
    cluster_to_main_channel = {c: np.random.randint(0, split_num) for c in range(n_clusters)}
    
    for idx, cluster_id in zip(semantic_indices, cluster_labels):
        main_channel = cluster_to_main_channel[cluster_id]
        
        # 判定是否触发“软划分/逃逸” (Softness)
        # 如果随机数 < softness，则该词不进入主通道，而是随机去其他通道
        if np.random.rand() < softness:
            # 随机选择一个非主通道
            choices = [x for x in range(split_num) if x != main_channel]
            target_channel = np.random.choice(choices)
        else:
            target_channel = main_channel
            
        channel_buckets[target_channel].append(idx)

    # --- 4. 构建最终 Shuffle ---
    # MCMark 的逻辑是：reshape(split_num, -1)。这意味着每个 bucket 的大小必须严格一致 (vocab_size // split_num)。
    # 但刚才的分配肯定是不均匀的。我们需要进行 "Rebalancing" (再平衡)。
    
    flat_shuffle = []
    target_len = vocab_size // split_num
    
    # 简单的再平衡策略：
    # 多的溢出到全局池，少的从全局池补。
    # 但为了保持语义，我们尽量只移动那些“高频/随机”的词，或者被迫移动语义词。
    
    overflow_pool = []
    final_buckets = [[] for _ in range(split_num)]
    
    # 第一轮：收集溢出
    for i in range(split_num):
        if len(channel_buckets[i]) > target_len:
            # 截断，多余的放入池子 (优先截断后加入的，即语义词，这稍微有点损耗，但可以接受)
            # 更好的做法是随机截断
            np.random.shuffle(channel_buckets[i])
            keep = channel_buckets[i][:target_len]
            overflow = channel_buckets[i][target_len:]
            final_buckets[i] = keep
            overflow_pool.extend(overflow)
        else:
            final_buckets[i] = channel_buckets[i] # 暂时保留，不够的后面补
            
    # 第二轮：填充不足
    np.random.shuffle(overflow_pool)
    pool_idx = 0
    
    for i in range(split_num):
        needed = target_len - len(final_buckets[i])
        if needed > 0:
            if pool_idx + needed > len(overflow_pool):
                # 极其罕见的情况：整除问题导致余数
                # 简单的填充剩下的
                fill = overflow_pool[pool_idx:]
            else:
                fill = overflow_pool[pool_idx : pool_idx + needed]
                pool_idx += needed
            final_buckets[i].extend(fill)
            
    # 展平
    final_indices = []
    for bucket in final_buckets:
        final_indices.extend(bucket)
        
    # 处理整除余数 (vocab_size % split_num != 0 的部分)
    # MCMark 代码中处理了余数，通常是通过截断或特殊逻辑。
    # 这里我们简单地把剩余的 overflow_pool 加到末尾，MCMark 会根据逻辑处理。
    remaining = vocab_size - len(final_indices)
    if remaining > 0:
        print(f"Warning: {remaining} tokens remaining due to divisibility. Appending to end.")
        # 找回还在 pool 里没用的
        leftover = overflow_pool[pool_idx:]
        # 如果还不够 (因为刚才的逻辑是按整除算的)，可能原本 vocab 就不能整除
        # 此时只需补上原 indices 中缺失的即可 (但这太复杂)
        # 简单方案：直接用 pool 剩下的
        final_indices.extend(leftover)

    # 转为 Tensor
    shuffle_tensor = torch.tensor(final_indices, dtype=torch.long)
    
    # 验证完整性
    assert len(torch.unique(shuffle_tensor)) == vocab_size, "Error: Duplicate or missing tokens in shuffle!"
    
    print(f"Saving shuffle tensor to {save_path}...")
    torch.save(shuffle_tensor, save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., facebook/opt-1.3b)")
    parser.add_argument("--split_num", type=int, default=4, help="Number of channels")
    parser.add_argument("--output", type=str, default="semantic_shuffle.pt")
    parser.add_argument("--softness", type=float, default=0.2, help="Probability of semantic leakage")
    args = parser.parse_args()
    
    generate_semantic_shuffle(args.model, args.split_num, args.output, args.softness)