#!/usr/bin/env python3
"""
分析堆滞后的具体问题
"""

import json

with open('merge_history_first_9.json', 'r') as f:
    history = json.load(f)

print("=" * 80)
print("堆滞后分析：为什么 'er' 被跳过")
print("=" * 80)

print("""
核心问题：当pair的count减少时，代码没有push新条目到堆中！

看tokenizer.py中的这段代码：
```python
if i > 0:
    old_near_pair = (old_tokens[i - 1], old_tokens[i])
    p2c[old_near_pair] -= w2c[word]  # count减少了！
    new_near_pair = (old_tokens[i - 1], new_token)
    p2w[new_near_pair].add(word)
    if word in p2w[old_near_pair]:
        p2w[old_near_pair].remove(word)
    p2c[new_near_pair] += w2c[word]
    heapq.heappush(heap, (-p2c[new_near_pair], ReverseSortPair(new_near_pair)))  # 只push了新pair！
```

注意：
1. `p2c[old_near_pair] -= w2c[word]` 减少了count
2. 但没有 `heapq.heappush` 来更新old_near_pair的条目
3. 堆中仍然是旧的count
4. 当这个pair被弹出时，`count != p2c.get(best_pair, 0)`，被跳过！
5. 更糟糕的是，如果没有其他操作push这个pair的新条目，它可能永远不会被选中！
""")

print("\n验证：在第6次merge (r,e) 中检查 'er' 的变化")
print("-" * 80)

for item in history:
    if item['type'] == 'merge' and item['merge_number'] == 6:
        print(f"第6次merge: (r, e) count={item['count']}")
        print("\n涉及 'er' 的变化:")
        
        # 查找所有涉及 e 或 r 的pair变化
        for pair_str, change in item['p2c_changes'].items():
            parts = pair_str.split(',')
            if len(parts) == 2:
                b1 = bytes.fromhex(parts[0]) if parts[0] else b''
                b2 = bytes.fromhex(parts[1]) if parts[1] else b''
                
                # 检查是否是 'er'
                if b1 == b'e' and b2 == b'r':
                    before = change['before']
                    after = change['after']
                    delta = after - before
                    print(f"  'er' ({pair_str}): {before} -> {after} (变化: {delta:+d})")
                    
                    if delta < 0:
                        print(f"    ⚠️  'er' 的count减少了{-delta}，但没有push新条目到堆！")
                        print(f"    堆中可能还有旧count={before}的条目")

print("\n" + "=" * 80)
print("关键证据：被跳过的过期条目")
print("=" * 80)

# 统计第6次之后被跳过的'er'条目
er_skips = []
for item in history:
    if item['type'] == 'skip_stale':
        pair = item['pair']
        if pair == ['65', '72']:  # e=65, r=72
            er_skips.append(item)

print(f"\n被跳过的 'er' 过期条目数量: {len(er_skips)}")
for skip in er_skips:
    print(f"  heap_count={skip['heap_count']}, current={skip['current_count']}")

print("""
这就是问题：
1. 第6次merge后，'er' count应该是1190
2. 但堆中可能还有旧条目（count=1300或1564）
3. 当这些过期条目被弹出时，会被跳过
4. 但关键是：如果代码没有正确push (er, 1190)，那么'er'就会从堆中消失！

让我们验证堆中是否真的有 (er, 1190)...
""")

print("\n" + "=" * 80)
print("验证：堆中 'er' 的条目数量")
print("=" * 80)

# 从第6次merge后的状态重建堆
for item in history:
    if item['type'] == 'merge' and item['merge_number'] == 6:
        p2c_after = {}
        for pair_str, change in item['p2c_changes'].items():
            parts = pair_str.split(',')
            if len(parts) == 2:
                b1 = bytes.fromhex(parts[0])
                b2 = bytes.fromhex(parts[1])
                p2c_after[(b1, b2)] = change['after']
        
        # 统计 'er' 的count
        er_count = p2c_after.get((b'e', b'r'), 0)
        print(f"第6次merge后，p2c中 'er' 的count = {er_count}")
        
        # 此时堆中应该有多少个 'er' 条目？
        # 理论上应该只有1个：(er, 1190)
        # 但实际上可能有多个过期条目
        
        print(f"\n但问题是：在第6次merge过程中，代码会:")
        print(f"  1. 多次减少 'er' 的count（通过 p2c[er] -= w2c[word]）")
        print(f"  2. 但不会push新的条目到堆中")
        print(f"  3. 只有当 'er' 作为 new_near_pair 被形成时，才会push")
        
        break

print("""
结论：
- 这不是"堆滞后"的简单问题
- 而是当count减少时，没有主动更新堆
- 这导致pair的有效条目可能丢失，或者被埋在过期条目之下
- 这就是为什么 'er' (count=1190) 没有排在堆顶的原因
""")
