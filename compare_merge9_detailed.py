#!/usr/bin/env python3
"""
详细对比第9次merge时的差异
"""

import json

# 加载我之前保存的p2c状态
with open('p2c_at_merge9_my_impl.json', 'r', encoding='utf-8') as f:
    my_data = json.load(f)

print("=" * 80)
print("我的实现 - 第9次merge前的状态分析")
print("=" * 80)

print("\n已完成的8次merges:")
for i, m in enumerate(my_data['merges_so_far']):
    # 将hex转换为bytes然后解码查看
    b1 = bytes.fromhex(m[0])
    b2 = bytes.fromhex(m[1])
    try:
        s1 = b1.decode('utf-8')
        s2 = b2.decode('utf-8')
    except:
        s1 = f"\\x{m[0]}"
        s2 = f"\\x{m[1]}"
    print(f"  {i+1}. ({s1}, {s2}) -> bytes: ({m[0]}, {m[1]})")

print(f"\n我的实现中即将进行第9次merge的pair:")
heap_top = my_data['heap_top']
b1 = bytes.fromhex(heap_top[0])
b2 = bytes.fromhex(heap_top[1])
try:
    s1 = b1.decode('utf-8')
    s2 = b2.decode('utf-8')
except:
    s1 = f"\\x{heap_top[0]}"
    s2 = f"\\x{heap_top[1]}"
print(f"  ({s1}, {s2}) count={my_data['heap_top_count']}")

# 查看 er 的count
print(f"\n在我的p2c中查找 'er':")
for k, v in my_data['p2c'].items():
    parts = k.split(',')
    if len(parts) == 2:
        b1 = bytes.fromhex(parts[0])
        b2 = bytes.fromhex(parts[1])
        if b1 == b'e' and b2 == b'r':
            print(f"  'er' count = {v}")

print("\n" + "=" * 80)
print("官方参考 - 第9次merge")
print("=" * 80)
print("官方第9次merge: ('e', 'r')")

print("\n" + "=" * 80)
print("关键问题分析")
print("=" * 80)
print("""
根据我的p2c数据:
- 'er' 的 count = 1190 (在我的p2c中)
- 'Ġs' 的 count = 1157 (在堆顶)

但是我的实现却选择了 'Ġs' 而不是 'er'。

可能的原因:
1. 'er' 在堆中有多个条目，堆顶的条目是过期的(stale)
2. 当从堆中弹出时，发现堆顶的count与p2c中当前count不匹配，所以跳过了
3. 这表明之前某个merge操作可能更新了'er'的count，但堆中的旧条目没有被正确移除

让我检查在之前的8次merges中，是否有操作会影响到 'er' 的count...
""")

# 分析哪些merge可能影响了 er
print("\n分析：哪些merge可能影响到 'er' pair:")
print("  'er' 是由 (b'e', b'r') 组成的")
print("  查看已完成的8次merges中是否有涉及 e 或 r 的合并:")

for i, m in enumerate(my_data['merges_so_far']):
    b1 = bytes.fromhex(m[0])
    b2 = bytes.fromhex(m[1])
    merged = b1 + b2
    if b'e' in [b1, b2] or b'r' in [b1, b2]:
        try:
            s1 = b1.decode('utf-8')
            s2 = b2.decode('utf-8')
        except:
            s1 = repr(b1)
            s2 = repr(b2)
        print(f"    第{i+1}次merge: ({s1}, {s2}) -> 可能影响 'er'")

print("""
问题可能出在:
- 当merge (b'h', b'e') 形成 'he' 时，可能会影响到 'er'，因为 'he' 后面可能跟着 'r'
- 或者当 merge (b'r', b'e') 形成 're' 时，可能会影响到 'er'

让我更仔细地检查 heap 的实现...
""")
