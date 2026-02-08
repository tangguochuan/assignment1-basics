# Tokenizer实现步骤
# 1. pretokenize: 将一个大的文本拆成多个chunk, 每个chunk并行
# 2. 对于每一个chunk, 执行OpenAI给的正则表达式进行分词
# 3. 维护一个数据结构，一个map: words -> count，eg. {"hello":5,...}
# 4. 维护一个数据结构，一个map: pair -> words eg. {(b'h',b'e'):["hello", "held"],...}
# 5. 维护一个数组结构，一个map: words -> tokens eg. {"hello": ['h','l','l','o','o']}
# 6. 维护一个最大堆，pair -> frequency
from typing import List, Tuple, Dict
import os
from multiprocessing import Pool,cpu_count
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time
import json
from collections import Counter,defaultdict
import heapq
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 将结果保存在json文件中
def save_tokens_json(tokens: List[str], output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

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
            word_encoded = word.encode("utf-8")
            l_list = [bytes([b]) for b in word_encoded]
            
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
        
def pre_token(file_path:str, num_chunks:int , special_tokens: List[str]):
    if os.path.exists(file_path) == False:
        raise FileNotFoundError(f"File not exists: {file_path}")
    with open(file_path, "rb") as f:
        chunks = find_chunk_boundaries(f, num_chunks, special_tokens)
        assert len(chunks), f"chunk length: {len(chunks)} is less than 1"
    # print(f"Len chunks: {len(chunks)}")
    params = [(file_path,start,end,special_tokens) for (start,end) in zip(chunks[:-1], chunks[1:])]
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
class ReverseSortPair:
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        # 频率相同时，字典序大的返回 True (即它更"小"，会排在堆顶)
        return self.pair > other.pair
    
def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    num_chunks = cpu_count() if cpu_count() !=0 else 8 
    pre_token_result_dict = pre_token(input_path, num_chunks,special_tokens)
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
    m = 0
    while cur_vocab_size < vocab_size:
        m += 1
        if not heap:
            break
        
        neg_count, wrapper = heapq.heappop(heap)
        best_pair = wrapper.pair
        count = - neg_count
        if count != p2c.get(best_pair,0):
            continue
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
                if i < len(old_tokens) -1 and (old_tokens[i], old_tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    if i > 0:
                        # (old_token[i-1], old_token[i]) 这对pair 的count减去一个word count
                        # p2w里面移除掉(old_token[i-1], old_token[i])对应的word
                        # 生成新pair: (old_token[i], new_token)
                        old_near_pair = (old_tokens[i - 1], old_tokens[i])
                        p2c[old_near_pair] -= w2c[word]
                        new_near_pair = (old_tokens[i -1], new_token)
                        p2w[new_near_pair].add(word)
                        if word in p2w[old_near_pair]:
                            p2w[old_near_pair].remove(word)
                        p2c[new_near_pair] += w2c[word]
                        heapq.heappush(heap,(-p2c[new_near_pair], ReverseSortPair(new_near_pair)))
                        if p2c[old_near_pair] > 0: 
                            heapq.heappush(heap, (-p2c[old_near_pair], ReverseSortPair(old_near_pair)))
                    if i < len(old_tokens) - 2:
                        if (old_tokens[i + 1], old_tokens[i + 2]) != best_pair:
                            old_near_pair = (old_tokens[i + 1], old_tokens[i + 2])
                            p2c[old_near_pair] -= w2c[word]
                            new_near_pair = (new_token, old_tokens[i + 2])
                            p2w[new_near_pair].add(word)
                            if word in p2w[old_near_pair]:
                                p2w[old_near_pair].remove(word)
                            p2c[new_near_pair] += w2c[word]
                            heapq.heappush(heap,(-p2c[new_near_pair], ReverseSortPair(new_near_pair)))
                            if p2c[old_near_pair] > 0: 
                                heapq.heappush(heap, (-p2c[old_near_pair], ReverseSortPair(old_near_pair)))
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            w2t[word] = new_tokens
        del p2w[best_pair]
        p2c[best_pair] = 0
    
    return vocab, merges
    

if __name__ == "__main__":
    vocab,merges = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    
