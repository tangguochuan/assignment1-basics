#!/usr/bin/env python3
"""
比较两个词汇表文件，找出差异
"""
import json

def load_vocab(path):
    """加载词汇表并返回 token->id 的字典"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_vocabs(vocab1_path, vocab2_path, name1="my-vocab", name2="reference"):
    """比较两个词汇表"""
    vocab1 = load_vocab(vocab1_path)
    vocab2 = load_vocab(vocab2_path)
    
    # 获取 token 集合
    tokens1 = set(vocab1.keys())
    tokens2 = set(vocab2.keys())
    
    # 仅在 vocab1 中的 token
    only_in_1 = tokens1 - tokens2
    # 仅在 vocab2 中的 token
    only_in_2 = tokens2 - tokens1
    # 两个词汇表中都有的 token
    common_tokens = tokens1 & tokens2
    
    # ID 不同的共同 token
    id_mismatch = []
    for token in common_tokens:
        if vocab1[token] != vocab2[token]:
            id_mismatch.append((token, vocab1[token], vocab2[token]))
    
    # 打印结果
    print("=" * 80)
    print(f"词汇表对比: {name1} vs {name2}")
    print("=" * 80)
    print()
    print(f"{name1} 总 token 数: {len(vocab1)}")
    print(f"{name2} 总 token 数: {len(vocab2)}")
    print(f"共同 token 数: {len(common_tokens)}")
    print()
    
    if only_in_1:
        print(f"仅在 {name1} 中有，{name2} 中没有的 token ({len(only_in_1)} 个):")
        print("-" * 60)
        # 按 ID 排序显示
        for token in sorted(only_in_1, key=lambda x: vocab1[x]):
            print(f"  '{token}' -> ID: {vocab1[token]}")
        print()
    else:
        print(f"✓ {name1} 中的所有 token 都在 {name2} 中")
        print()
    
    if only_in_2:
        print(f"仅在 {name2} 中有，{name1} 中没有的 token ({len(only_in_2)} 个):")
        print("-" * 60)
        # 按 ID 排序显示
        for token in sorted(only_in_2, key=lambda x: vocab2[x]):
            print(f"  '{token}' -> ID: {vocab2[token]}")
        print()
    else:
        print(f"✓ {name2} 中的所有 token 都在 {name1} 中")
        print()
    
    if id_mismatch:
        print(f"两个词汇表中都有，但 ID 不同的 token ({len(id_mismatch)} 个):")
        print("-" * 60)
        for token, id1, id2 in sorted(id_mismatch, key=lambda x: x[1]):
            print(f"  '{token}': {name1} ID={id1}, {name2} ID={id2}, 差异={id1-id2:+d}")
        print()
    else:
        print("✓ 所有共同 token 的 ID 都一致")
        print()
    
    # 总结
    print("=" * 80)
    print("总结:")
    if not only_in_1 and not only_in_2 and not id_mismatch:
        print("  ✓ 两个词汇表完全相同！")
    else:
        print(f"  - {name1} 独有的 token: {len(only_in_1)}")
        print(f"  - {name2} 独有的 token: {len(only_in_2)}")
        print(f"  - ID 不匹配的 token: {len(id_mismatch)}")
    print("=" * 80)

if __name__ == "__main__":
    compare_vocabs(
        "tests/fixtures/my-vocab.json",
        "tests/fixtures/train-bpe-reference-vocab.json",
        name1="my-vocab",
        name2="reference-vocab"
    )
