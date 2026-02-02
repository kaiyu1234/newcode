# tools/generate_semantic_shuffle.py

import torch
import argparse
import os
import numpy as np
from transformers import AutoModel
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import math

def generate_semantic_shuffle(model_name, split_num, save_path, high_freq_ratio=0.2, softness=0.2):
    # 1. 固定随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading model: {model_name}...")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    vocab_size = embeddings.shape[0]
    target_size = vocab_size // split_num # 最终每个通道的目标大小
    print(f"Vocab Size: {vocab_size}, Target Channel Size: {target_size}")

    # --- Step 1: 频率分层 ---
    norms = np.linalg.norm(embeddings, axis=1)
    sorted_indices = np.argsort(norms)
    
    cutoff = int(vocab_size * high_freq_ratio)
    high_freq_indices = sorted_indices[:cutoff].copy()
    np.random.shuffle(high_freq_indices)
    
    semantic_indices = sorted_indices[cutoff:].copy()
    n_semantic = len(semantic_indices)
    
    print(f"High Freq Tokens: {len(high_freq_indices)}")
    print(f"Semantic Tokens:  {n_semantic}")

    # --- Step 2: 语义聚类 ---
    print("Clustering semantic tokens...")
    semantic_embeddings = embeddings[semantic_indices]
    
    # 假设每个通道分 100 个簇
    clusters_per_channel = 100
    n_clusters = split_num * clusters_per_channel
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init="auto")
    cluster_labels = kmeans.fit_predict(semantic_embeddings)
    cluster_centers = kmeans.cluster_centers_

    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in zip(semantic_indices, cluster_labels):
        clusters[label].append(idx)

    # --- Step 3: 计算标准簇大小 (Standard Cluster Size) ---
    # 我们希望簇尽可能均匀，这样 Softness 带来的波动也会比较均匀
    target_cluster_size = math.ceil(n_semantic / n_clusters)
    print(f"Target Cluster Size: {target_cluster_size}")

    # --- Step 4: 强制剪枝 (Pruning Oversized Clusters) ---
    print("Pruning oversized clusters...")
    pool_indices = []
    underfilled_clusters = [] 
    
    for cid in range(n_clusters):
        members = clusters[cid]
        curr_len = len(members)
        
        if curr_len > target_cluster_size:
            # 距离排序
            member_embs = embeddings[members]
            center = cluster_centers[cid:cid+1]
            dists = cdist(member_embs, center, metric='cosine').squeeze()
            if dists.shape == (): dists = np.array([dists])
            sorted_arg = np.argsort(dists)
            
            # 截断
            keep_idx = [members[i] for i in sorted_arg[:target_cluster_size]]
            drop_idx = [members[i] for i in sorted_arg[target_cluster_size:]]
            
            clusters[cid] = keep_idx
            pool_indices.extend(drop_idx)
            
        elif curr_len < target_cluster_size:
            underfilled_clusters.append(cid)
            
    print(f"Pool size after pruning: {len(pool_indices)}")

    # --- Step 5: 受限回填 (Constrained Refill) ---
    if len(pool_indices) > 0:
        print("Refilling pool to underfilled clusters...")
        pool_arr = np.array(pool_indices)
        pool_embs = embeddings[pool_arr]
        dists_matrix = cdist(pool_embs, cluster_centers, metric='cosine')
        
        capacity_left = np.array([target_cluster_size - len(clusters[i]) for i in range(n_clusters)])
        
        for i, token_idx in enumerate(pool_indices):
            dists = dists_matrix[i]
            sorted_clusters_idx = np.argsort(dists)
            
            found = False
            for cid in sorted_clusters_idx:
                if capacity_left[cid] > 0:
                    clusters[cid].append(token_idx)
                    capacity_left[cid] -= 1
                    found = True
                    break
            
            if not found:
                # 兜底：如果实在没地儿去了，强行塞给最近的
                best_cid = sorted_clusters_idx[0]
                clusters[best_cid].append(token_idx)

    # --- Step 6: 分配通道 + 软化逃逸 (Allocation with Softness) ---
    print(f"Assigning to channels with Softness = {softness}...")
    
    channel_buckets = [[] for _ in range(split_num)]
    
    # 6.1 高频词 (本来就是随机的，直接分配)
    hf_chunk = len(high_freq_indices) // split_num
    for i in range(split_num):
        start = i * hf_chunk
        end = (i + 1) * hf_chunk if i != split_num - 1 else len(high_freq_indices)
        channel_buckets[i].extend(high_freq_indices[start:end])
    
    # 6.2 语义簇 (带 Softness)
    cluster_ids = np.arange(n_clusters)
    np.random.shuffle(cluster_ids) # 随机打乱簇
    
    clusters_per_split = len(cluster_ids) // split_num
    
    for i in range(split_num):
        # 确定本通道负责的簇 (Main Channel = i)
        start = i * clusters_per_split
        end = (i + 1) * clusters_per_split if i != split_num - 1 else len(cluster_ids)
        assigned_cids = cluster_ids[start:end]
        
        main_channel = i
        
        for cid in assigned_cids:
            tokens = clusters[cid]
            for token in tokens:
                # [关键] 软化逻辑
                if np.random.rand() < softness:
                    # 逃逸：随机去任意通道
                    target_ch = np.random.randint(0, split_num)
                else:
                    # 归队：去主通道
                    target_ch = main_channel
                
                channel_buckets[target_ch].append(token)

    # --- Step 7: 最终强制平衡 (Final Rebalancing) ---
    # 因为 Softness 的存在，现在 channel_buckets 的大小肯定是不均匀的
    # 我们必须进行“削峰填谷”，这对于 Unbiased Watermark 至关重要
    
    print("Executing Final Rebalancing (correcting softness deviations)...")
    
    global_overflow = []
    
    # Phase 1: 削峰 (对溢出的通道，截断多余部分)
    # 既然已经有了 Softness，这里直接截断尾部即可，无需复杂计算
    for i in range(split_num):
        curr = channel_buckets[i]
        if len(curr) > target_size:
            keep = curr[:target_size]
            overflow = curr[target_size:]
            channel_buckets[i] = keep
            global_overflow.extend(overflow)
            
    # Phase 2: 填谷 (对亏损的通道，用溢出词填充)
    for i in range(split_num):
        needed = target_size - len(channel_buckets[i])
        if needed > 0:
            if len(global_overflow) >= needed:
                fill = global_overflow[:needed]
                channel_buckets[i].extend(fill)
                global_overflow = global_overflow[needed:]
            else:
                # 极少数情况：溢出不够填 (因为 target_size 取整误差)
                # 从已满的通道里“借”或者复制？通常不会发生，除非 vocab_size 很小
                # 简单策略：循环填充
                print(f"Warning: Overflow pool exhausted for channel {i}")
                if len(channel_buckets[i]) > 0: 
                    # 自体复制填充 (虽然有点脏，但能保证运行)
                    channel_buckets[i].extend(channel_buckets[i][:needed])

    # 处理剩余的 overflow (通常是取整余数) -> 塞给最后一个通道
    if len(global_overflow) > 0:
        channel_buckets[-1].extend(global_overflow)

    # --- Step 8: 保存与验证 ---
    final_flat = []
    final_counts = [len(b) for b in channel_buckets]
    print(f"Final Channel Counts: {final_counts}") # 应该全都是 target_size (或极接近)
    
    for b in channel_buckets:
        final_flat.extend(b)
        
    shuffle_tensor = torch.tensor(final_flat, dtype=torch.long)
    
    # 尺寸修正
    if len(shuffle_tensor) > vocab_size:
        shuffle_tensor = shuffle_tensor[:vocab_size]
    elif len(shuffle_tensor) < vocab_size:
        diff = vocab_size - len(shuffle_tensor)
        shuffle_tensor = torch.cat([shuffle_tensor, torch.zeros(diff, dtype=torch.long)])
    
    print(f"Saving to {save_path}...")
    torch.save(shuffle_tensor, save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split_num", type=int, default=4)
    parser.add_argument("--output", type=str, default="semantic_shuffle.pt")
    parser.add_argument("--high_freq_ratio", type=float, default=0.2)
    parser.add_argument("--softness", type=float, default=0.2, help="Escape probability")
    args = parser.parse_args()
    
    generate_semantic_shuffle(args.model, args.split_num, args.output, args.high_freq_ratio, args.softness)