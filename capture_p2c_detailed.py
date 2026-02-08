#!/usr/bin/env python3
"""
更详细地捕获第9次merge时的状态
"""

import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import pre_token, ReverseSortPair
from collections import Counter, defaultdict
import heapq
from typing import List, Tuple, Dict
import copy

def train_bpe_with_detailed_capture(input_path: str | os.PathLike, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]], dict]:
    """
    修改版的train_bpe，详细记录每次merge前后的状态
    """
    from multiprocessing import cpu_count
    
    num_chunks = cpu_count() if cpu_count() != 0 else 8 
    pre_token_result_dict = pre_token(input_path, num_chunks, special_tokens)
    w2c = pre_token_result_dict["words_to_count"]
    w2t = pre_token_result_dict["words_to_tokens"]
    p2w = pre_token_result_dict["pair_to_words"]
    p2c = pre_token_result_dict["pair_to_count"]
    
    # 先放special token（从ID 0开始），然后是256个字节
    vocab = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])
    cur_vocab_size = 256 + len(special_tokens)
    
    heap = [(-count, ReverseSortPair(pair)) for pair, count in p2c.items() if count > 0]
    heapq.heapify(heap)
    merges = []
    
    merge_history = []
    
    while cur_vocab_size < vocab_size:
        if not heap:
            break
        
        neg_count, wrapper = heapq.heappop(heap)
        best_pair = wrapper.pair
        count = - neg_count
        if count != p2c.get(best_pair, 0):
            # 过期条目，记录
            merge_history.append({
                'type': 'skip_stale',
                'pair': (best_pair[0].hex(), best_pair[1].hex()),
                'heap_count': count,
                'current_count': p2c.get(best_pair, 0)
            })
            continue
        
        # 记录这次有效的merge
        merge_info = {
            'type': 'merge',
            'merge_number': len(merges) + 1,
            'pair': (best_pair[0].hex(), best_pair[1].hex()),
            'count': count,
            'p2c_before': dict(p2c),  # 记录merge前的p2c
            'heap_size': len(heap),
        }
        
        merges.append(best_pair)
        words_to_update = list(p2w[best_pair])
        new_token = best_pair[0] + best_pair[1]
        vocab[cur_vocab_size] = new_token
        cur_vocab_size += 1
        
        for word in words_to_update:
            new_tokens = []
            old_tokens = w2t[word]
            i = 0
            while i < len(old_tokens):
                if i < len(old_tokens) - 1 and (old_tokens[i], old_tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    if i > 0:
                        old_near_pair = (old_tokens[i - 1], old_tokens[i])
                        p2c[old_near_pair] -= w2c[word]
                        new_near_pair = (old_tokens[i - 1], new_token)
                        p2w[new_near_pair].add(word)
                        if word in p2w[old_near_pair]:
                            p2w[old_near_pair].remove(word)
                        p2c[new_near_pair] += w2c[word]
                        heapq.heappush(heap, (-p2c[new_near_pair], ReverseSortPair(new_near_pair)))
                    if i < len(old_tokens) - 2:
                        if (old_tokens[i + 1], old_tokens[i + 2]) != best_pair:
                            old_near_pair = (old_tokens[i + 1], old_tokens[i + 2])
                            p2c[old_near_pair] -= w2c[word]
                            new_near_pair = (new_token, old_tokens[i + 2])
                            p2w[new_near_pair].add(word)
                            if word in p2w[old_near_pair]:
                                p2w[old_near_pair].remove(word)
                            p2c[new_near_pair] += w2c[word]
                            heapq.heappush(heap, (-p2c[new_near_pair], ReverseSortPair(new_near_pair)))
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            w2t[word] = new_tokens
        del p2w[best_pair]
        p2c[best_pair] = 0
        
        merge_info['p2c_after'] = dict(p2c)  # 记录merge后的p2c
        merge_history.append(merge_info)
        
        # 在第9次merge后停止
        if len(merges) == 9:
            break
    
    return vocab, merges, merge_history


def main():
    input_path = "tests/fixtures/corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    
    print("运行BPE训练并详细记录前9次merge...")
    vocab, merges, history = train_bpe_with_detailed_capture(input_path, vocab_size, special_tokens)
    
    # 保存详细历史
    serializable_history = []
    for item in history:
        s_item = {
            'type': item['type'],
        }
        if item['type'] == 'merge':
            s_item['merge_number'] = item['merge_number']
            s_item['pair'] = item['pair']
            s_item['count'] = item['count']
            s_item['heap_size'] = item['heap_size']
            # 不保存完整的p2c，只保存关键pair的count变化
            pair_hex = item['pair']
            p2c_before = item['p2c_before']
            p2c_after = item['p2c_after']
            
            # 找出count发生变化的pair
            changes = {}
            for k, v in p2c_after.items():
                before_v = p2c_before.get(k, 0)
                if v != before_v:
                    changes[f"{k[0].hex()},{k[1].hex()}"] = {'before': before_v, 'after': v}
            s_item['p2c_changes'] = changes
        else:
            s_item['pair'] = item['pair']
            s_item['heap_count'] = item['heap_count']
            s_item['current_count'] = item['current_count']
        serializable_history.append(s_item)
    
    with open('merge_history_first_9.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=2, ensure_ascii=False)
    
    print("\n前9次merge的详细历史已保存到 merge_history_first_9.json")
    
    # 打印关键信息
    print("\n" + "=" * 80)
    print("前9次merge摘要")
    print("=" * 80)
    
    skip_count = 0
    for item in history:
        if item['type'] == 'skip_stale':
            skip_count += 1
            if skip_count <= 5:  # 只打印前5个跳过的
                b1 = bytes.fromhex(item['pair'][0])
                b2 = bytes.fromhex(item['pair'][1])
                print(f"  [跳过] {b1}, {b2}: heap_count={item['heap_count']}, current_count={item['current_count']}")
        elif item['type'] == 'merge':
            b1 = bytes.fromhex(item['pair'][0])
            b2 = bytes.fromhex(item['pair'][1])
            print(f"  [Merge {item['merge_number']}] {b1}, {b2} (count={item['count']})")
    
    print(f"\n总共跳过了 {skip_count} 个过期条目")
    
    # 特别关注第6次merge (r,e) 对 'er' 的影响
    print("\n" + "=" * 80)
    print("检查第6次merge (r,e) 对 'er' 的影响")
    print("=" * 80)
    
    for item in history:
        if item['type'] == 'merge' and item['merge_number'] == 6:
            changes = item.get('p2c_changes', {})
            for pair_str, change in changes.items():
                if '65,72' in pair_str:  # e=65(hex), r=72(hex)
                    print(f"  'er' count变化: {change['before']} -> {change['after']}")


if __name__ == "__main__":
    main()
