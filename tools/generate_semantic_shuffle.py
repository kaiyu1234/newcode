# tools/generate_semantic_shuffle.py

import torch
import argparse
import os
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

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
    
    # indices = np.arange(vocab_size)
    # np.random.shuffle(indices) # 先全打乱
    
    # cutoff = int(vocab_size * high_freq_ratio)
    # high_freq_indices = indices[:cutoff] # 这部分将保持完全随机 (High Entropy)
    # semantic_indices = indices[cutoff:]  # 这部分将进行聚类 (Semantic Preservation)
    
    # print(f"Processing: {len(high_freq_indices)} random tokens, {len(semantic_indices)} semantic tokens.")



    # ================= 修改开始 =================
    # --- 1. 频率分层 (基于 Embedding Norm 的启发式策略) ---
    print("Calculating Embedding Norms to estimate token frequency...")
    
    # 计算每个 Token Embedding 的 L2 范数 (模长)
    # axis=1 表示对每个 (hidden_size,) 的向量计算范数
    # 原理：高频词/功能词在训练中更新频繁，Embedding Norm 通常较小 (靠近原点)
    norms = np.linalg.norm(embeddings, axis=1)
    
    # 对 Norm 进行从小到大排序，返回索引
    # sorted_indices[0] 是 Norm 最小的词 (大概率是 "the", ",", "." 等)
    sorted_indices = np.argsort(norms)
    
    cutoff = int(vocab_size * high_freq_ratio)
    
    # A. 选取 Norm 最小的前 N% 作为“高熵/高频组”
    # 这些词虽然是被 Norm 选出来的，但在分配给 Channel 时需要保持“随机均匀性”
    high_freq_indices = sorted_indices[:cutoff].copy()
    
    # [关键步骤] 必须对 high_freq_indices 内部进行一次打乱！
    # 如果不打乱，Channel 0 会分到 Norm 极小的词，Channel k 会分到 Norm 稍大的词，导致通道间特征不平衡。
    np.random.shuffle(high_freq_indices)
    
    # B. 剩余的 (Norm 较大的) 作为“语义组”，用于后续聚类
    semantic_indices = sorted_indices[cutoff:].copy()
    
    print(f"Heuristic Partition: {len(high_freq_indices)} tokens (Low Norm) -> Random Shuffle")
    print(f"                     {len(semantic_indices)} tokens (High Norm) -> Semantic Clustering")
    # ================= 修改结束 =================
    

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
    
    # --- 4. 构建最终 Shuffle (基于语义最优传输方案) ---
    print("Rebalancing buckets using Semantic-Optimal Strategy...")
    
    target_len = vocab_size // split_num
    final_buckets = [None] * split_num
    overflow_pool = []
    
    # 辅助函数：获取一个 bucket 中所有 token 的 embeddings
    def get_bucket_embeddings(bucket_indices):
        return embeddings[bucket_indices]

    # --- Phase 1: 处理溢出 (Pruning Overflows) ---
    # 策略：保留最核心的语义词，把边缘词扔出去
    for i in range(split_num):
        current_indices = channel_buckets[i]
        current_len = len(current_indices)
        
        if current_len > target_len:
            # 计算该通道的语义中心 (Centroid)
            # 注意：如果 bucket 里大部分是随机分配的高频词，中心可能没意义，但这是目前最优解
            bucket_embs = get_bucket_embeddings(current_indices)
            centroid = np.mean(bucket_embs, axis=0, keepdims=True)
            
            # 计算每个词到中心的距离 (使用 Cosine Distance)
            dists = cdist(bucket_embs, centroid, metric='cosine').squeeze()
            
            # 按距离排序 (从小到大)
            sorted_arg = np.argsort(dists)
            
            # 保留最近的 target_len 个 (Keep the core semantics)
            keep_arg = sorted_arg[:target_len]
            overflow_arg = sorted_arg[target_len:]
            
            # 转换回原始 index
            keep_indices = [current_indices[k] for k in keep_arg]
            overflow_indices = [current_indices[k] for k in overflow_arg]
            
            final_buckets[i] = keep_indices
            overflow_pool.extend(overflow_indices)
        else:
            # 暂时不够，先不动，等下一轮填
            final_buckets[i] = current_indices

    print(f"Overflow pool size: {len(overflow_pool)}")

    # --- Phase 2: 填充缺口 (Filling Underflows) ---
    # 策略：从池子中选距离缺口通道中心最近的词填入
    
    # 先把 overflow_pool 转成 numpy 以便快速计算
    if len(overflow_pool) > 0:
        pool_indices = np.array(overflow_pool)
        pool_embeddings = get_bucket_embeddings(pool_indices)
        pool_mask = np.ones(len(pool_indices), dtype=bool) # 标记是否已被使用
    
    for i in range(split_num):
        current_indices = final_buckets[i]
        needed = target_len - len(current_indices)
        
        if needed > 0:
            if needed > np.sum(pool_mask):
                print(f"Warning: Not enough tokens in pool for channel {i}!")
                # 极其罕见，只能跳过或报错
                break
                
            # 计算当前通道的语义中心
            # 如果通道是空的(极其罕见)，就用全局中心或者随机选一个
            if len(current_indices) > 0:
                bucket_embs = get_bucket_embeddings(current_indices)
                centroid = np.mean(bucket_embs, axis=0, keepdims=True)
            else:
                centroid = np.mean(embeddings, axis=0, keepdims=True)
            
            # 只计算还没被用掉的 pool token 的距离
            active_pool_idx = np.where(pool_mask)[0]
            active_pool_embs = pool_embeddings[active_pool_idx]
            
            # 计算距离
            dists = cdist(active_pool_embs, centroid, metric='cosine').squeeze()
            
            # 找最近的 needed 个
            # 注意：这里是用 argpartition 或 argsort
            if needed < len(dists):
                nearest_arg_in_active = np.argpartition(dists, needed)[:needed]
            else:
                nearest_arg_in_active = np.arange(len(dists))
                
            # 获取在 pool_indices 中的真实索引
            real_pool_indices_idx = active_pool_idx[nearest_arg_in_active]
            
            # 选出的 token ID
            selected_tokens = pool_indices[real_pool_indices_idx].tolist()
            
            # 填入 bucket
            final_buckets[i].extend(selected_tokens)
            
            # 从 pool 中标记为已用
            pool_mask[real_pool_indices_idx] = False

    # --- Phase 3: 最终组装 ---
    final_indices = []
    for bucket in final_buckets:
        final_indices.extend(bucket)
        
    # 处理无法整除的剩余部分 (如果有的话，直接加到末尾，和原逻辑一致)
    # 检查 pool 里还有没有没用的
    if len(overflow_pool) > 0:
        remaining_in_pool = pool_indices[pool_mask].tolist()
        if len(remaining_in_pool) > 0:
             # 只有当 vocab_size % split_num != 0 时才会发生
             final_indices.extend(remaining_in_pool)

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