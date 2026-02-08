# Tokenizer实现 - 修复版（正确处理 Ġ）
from typing import List, Tuple, Dict
import os
from multiprocessing import Pool, cpu_count
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import json
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# GPT-2 bytes_to_unicode 映射表
# 将字节 0-255 映射到可打印的 Unicode 字符
BYTES_TO_UNICODE = {
    n: chr(c) 
    for n, c in [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
        (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
        (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23),
        (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30), (31, 31),
        (32, 32), (33, 33), (34, 34), (35, 35), (36, 36), (37, 37), (38, 38), (39, 39),
        (40, 40), (41, 41), (42, 42), (43, 43), (44, 44), (45, 45), (46, 46), (47, 47),
        (48, 48), (49, 49), (50, 50), (51, 51), (52, 52), (53, 53), (54, 54), (55, 55),
        (56, 56), (57, 57), (58, 58), (59, 59), (60, 60), (61, 61), (62, 62), (63, 63),
        (64, 64), (65, 65), (66, 66), (67, 67), (68, 68), (69, 69), (70, 70), (71, 71),
        (72, 72), (73, 73), (74, 74), (75, 75), (76, 76), (77, 77), (78, 78), (79, 79),
        (80, 80), (81, 81), (82, 82), (83, 83), (84, 84), (85, 85), (86, 86), (87, 87),
        (88, 88), (89, 89), (90, 90), (91, 91), (92, 92), (93, 93), (94, 94), (95, 95),
        (96, 96), (97, 97), (98, 98), (99, 99), (100, 100), (101, 101), (102, 102), (103, 103),
        (104, 104), (105, 105), (106, 106), (107, 107), (108, 108), (109, 109), (110, 110), (111, 111),
        (112, 112), (113, 113), (114, 114), (115, 115), (116, 116), (117, 117), (118, 118), (119, 119),
        (120, 120), (121, 121), (122, 122), (123, 123), (124, 124), (125, 125), (126, 126), (127, 127),
        (128, 128), (129, 129), (130, 130), (131, 131), (132, 132), (133, 133), (134, 134), (135, 135),
        (136, 136), (137, 137), (138, 138), (139, 139), (140, 140), (141, 141), (142, 142), (143, 143),
        (144, 144), (145, 145), (146, 146), (147, 147), (148, 148), (149, 149), (150, 150), (151, 151),
        (152, 152), (153, 153), (154, 154), (155, 155), (156, 156), (157, 157), (158, 158), (159, 159),
        (160, 160), (161, 161), (162, 162), (163, 163), (164, 164), (165, 165), (166, 166), (167, 167),
        (168, 168), (169, 169), (170, 170), (171, 171), (172, 172), (173, 173), (174, 174), (175, 175),
        (176, 176), (177, 177), (178, 178), (179, 179), (180, 180), (181, 181), (182, 182), (183, 183),
        (184, 184), (185, 185), (186, 186), (187, 187), (188, 188), (189, 189), (190, 190), (191, 191),
        (192, 192), (193, 193), (194, 194), (195, 195), (196, 196), (197, 197), (198, 198), (199, 199),
        (200, 200), (201, 201), (202, 202), (203, 203), (204, 204), (205, 205), (206, 206), (207, 207),
        (208, 208), (209, 209), (210, 210), (211, 211), (212, 212), (213, 213), (214, 214), (215, 215),
        (216, 216), (217, 217), (218, 218), (219, 219), (220, 220), (221, 221), (222, 222), (223, 223),
        (224, 224), (225, 225), (226, 226), (227, 227), (228, 228), (229, 229), (230, 230), (231, 231),
        (232, 232), (233, 233), (234, 234), (235, 235), (236, 236), (237, 237), (238, 238), (239, 239),
        (240, 240), (241, 241), (242, 242), (243, 243), (244, 244), (245, 245), (246, 246), (247, 247),
        (248, 248), (249, 249), (250, 250), (251, 251), (252, 252), (253, 253), (254, 254), (255, 255),
    ]
}
# 实际上 GPT-2 使用不同的映射，把不可打印字节映射到特定的 Unicode 字符
# 但为了简单，我们可以直接使用 bytes 作为 key

def process_single_chunk(file_path, start, end, special_tokens: List[str]):
    words_to_count = Counter()
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = '|'.join(re.escape(token) for token in sorted_tokens)
        chunks = re.split(special_pattern, text)
        for chunk in chunks:
            if not chunk:
                continue

            tokens = re.findall(PAT, chunk)
            for t in tokens:
                words_to_count[t] += 1
    
    words_to_tokens = {}
    pair_to_words = defaultdict(set)
    pair_to_count = Counter()
    for word, count in words_to_count.items():
        # 关键修复：将前导空格替换为 Ġ
        if word.startswith(' '):
            word = 'Ġ' + word[1:]
        
        # 将每个字符（包括多字节字符如 Ġ）作为独立的 token
        # 而不是将 UTF-8 字节拆分
        l_list = [c.encode("utf-8") for c in word]
        
        words_to_tokens[word] = l_list
        if len(l_list) >= 2:
            for i in range(len(l_list) - 1):
                pair = (l_list[i], l_list[i + 1])
                pair_to_words[pair].add(word)
                pair_to_count[pair] += count
    return {
        "words_to_count": words_to_count,
        "words_to_tokens": words_to_tokens,
        "pair_to_words": pair_to_words,
        "pair_to_count": pair_to_count
    }


def pre_token(file_path: str, num_chunks: int, special_tokens: List[str]):
    if os.path.exists(file_path) == False:
        raise FileNotFoundError(f"File not exists: {file_path}")
    with open(file_path, "rb") as f:
        chunks = find_chunk_boundaries(f, num_chunks, special_tokens)
        assert len(chunks), f"chunk length: {len(chunks)} is less than 1"
    
    params = [(file_path, start, end, special_tokens) for (start, end) in zip(chunks[:-1], chunks[1:])]
    chunk_results = []
    with Pool(len(chunks) - 1) as pool:
        chunk_results = pool.starmap(process_single_chunk, params)
    
    words_to_count = Counter()
    pair_to_count = Counter()
    words_to_tokens = {}
    pair_to_words = defaultdict(set)
    for chunk_result in chunk_results:
        words_to_count.update(chunk_result["words_to_count"])
        pair_to_count.update(chunk_result["pair_to_count"])
        words_to_tokens.update(chunk_result["words_to_tokens"])
        for pair, word_set in chunk_result["pair_to_words"].items():
            pair_to_words[pair].update(word_set)

    return {
        "words_to_count": words_to_count,
        "words_to_tokens": words_to_tokens,
        "pair_to_words": pair_to_words,
        "pair_to_count": pair_to_count
    }


def get_best_pair(p2c):
    """从 p2c 中找到频率最大的 pair。频率相同则字典序大的优先。"""
    if not p2c:
        return None, 0
    
    max_count = 0
    best_pair = None
    
    for pair, count in p2c.items():
        if count > max_count:
            max_count = count
            best_pair = pair
        elif count == max_count and count > 0:
            if pair > best_pair:
                best_pair = pair
    
    if max_count == 0:
        return None, 0
    
    return best_pair, max_count


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    num_chunks = cpu_count() if cpu_count() != 0 else 8 
    pre_token_result_dict = pre_token(input_path, num_chunks, special_tokens)
    w2c = pre_token_result_dict["words_to_count"]
    w2t = pre_token_result_dict["words_to_tokens"]
    p2w = pre_token_result_dict["pair_to_words"]
    p2c = pre_token_result_dict["pair_to_count"]
    
    # 先放 special token（从ID 0开始）
    vocab = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    
    # 然后放基础字符（256个字节对应的字符）
    # 注意：这里需要按照 Ġ 的编码来放
    for i in range(256):
        if i == 0xc4:
            # 0xc4 是 Ġ 的第一个字节，但我们需要放完整的 Ġ
            continue
        elif i == 0xa0:
            # 放完整的 Ġ
            vocab[len(special_tokens) + i - 1] = 'Ġ'.encode("utf-8")  # ID = special_len + 0xa0 - 1
        else:
            # 普通字节
            vocab[len(special_tokens) + i] = bytes([i])
    
    # 重新调整 vocab ID
    # 更简单的方式：先收集所有基础字符，然后分配 ID
    vocab = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    
    # 收集所有在数据中出现的单字节字符
    base_chars = set()
    for word, tokens in w2t.items():
        for t in tokens:
            if len(t) == 1:  # 单字节
                base_chars.add(t)
            else:  # 多字节（如 Ġ）
                base_chars.add(t)
    
    # 按照一定顺序分配 ID
    # 先放 Ġ，然后放其他字符
    sorted_chars = sorted(base_chars, key=lambda x: (x != b'\xc4\xa0', x))
    for i, char in enumerate(sorted_chars):
        vocab[len(special_tokens) + i] = char
    
    cur_vocab_size = len(vocab)
    merges = []
    m = 0
    
    while cur_vocab_size < vocab_size:
        m += 1
        
        best_pair, count = get_best_pair(p2c)
        
        if best_pair is None or count == 0:
            break
        
        # 调试输出
        if m <= 45:
            b1, b2 = best_pair
            print(f"merge {m:2d}: count={count:4d}, pair=({b1!r}, {b2!r})")
        
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
                    
                    if i < len(old_tokens) - 2:
                        if (old_tokens[i + 1], old_tokens[i + 2]) != best_pair:
                            old_near_pair = (old_tokens[i + 1], old_tokens[i + 2])
                            p2c[old_near_pair] -= w2c[word]
                            new_near_pair = (new_token, old_tokens[i + 2])
                            p2w[new_near_pair].add(word)
                            if word in p2w[old_near_pair]:
                                p2w[old_near_pair].remove(word)
                            p2c[new_near_pair] += w2c[word]
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            w2t[word] = new_tokens
        
        del p2w[best_pair]
        p2c[best_pair] = 0
    
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    print(f"\n总共生成 {len(merges)} 个 merges")
    
    # 保存 merges
    with open("my-merges-fixed.txt", "wb") as f:
        for pair in merges:
            b1, b2 = pair
            f.write(b1 + b" " + b2 + b"\n")
    print(f"已保存到 my-merges-fixed.txt")
