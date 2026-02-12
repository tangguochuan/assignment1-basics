# Tokenizer实现步骤
# 1. pretokenize: 将一个大的文本拆成多个chunk, 每个chunk并行
# 2. 对于每一个chunk, 执行OpenAI给的正则表达式进行分词
# 3. 维护一个数据结构，一个map: words -> count，eg. {"hello":5,...}
# 4. 维护一个数据结构，一个map: pair -> words eg. {(b'h',b'e'):["hello", "held"],...}
# 5. 维护一个数组结构，一个map: words -> tokens eg. {"hello": ['h','l','l','o','o']}
# 6. 维护一个最大堆，pair -> frequency
from typing import List, Tuple, Dict, Iterable, Iterator
import os
from multiprocessing import Pool,cpu_count
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import json
import base64
from pathlib import Path
from collections import Counter,defaultdict
import heapq
import time
import tracemalloc
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
        return self.pair > other.pair
    
def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 开始记录内存和时间
    tracemalloc.start()
    start_time = time.time()
    
    num_chunks = cpu_count() if cpu_count() !=0 else 8 
    pre_token_result_dict = pre_token(input_path, num_chunks,special_tokens)
    w2c = pre_token_result_dict["words_to_count"]
    w2t = pre_token_result_dict["words_to_tokens"]
    p2w = pre_token_result_dict["pair_to_words"]
    p2c = pre_token_result_dict["pair_to_count"]
    
    vocab = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])
    cur_vocab_size = 256 + len(special_tokens)
    heap = [(-count, ReverseSortPair(pair)) for pair, count in p2c.items() if count > 0]
    heapq.heapify(heap)
    merges = []
    while cur_vocab_size < vocab_size:
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
        pair_count_changes = defaultdict(int)
        for word in words_to_update:
            new_tokens = []
            old_tokens = w2t[word]
            i = 0
            word_count = w2c[word]
            while i < len(old_tokens):
                if i < len(old_tokens) -1 and (old_tokens[i], old_tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            for i in range(len(old_tokens) - 1):
                old_pair = (old_tokens[i], old_tokens[i + 1])
                pair_count_changes[old_pair] -= word_count
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                pair_count_changes[new_pair] += word_count
                p2w[new_pair].add(word)
            w2t[word] = new_tokens
        for pair,change in pair_count_changes.items():
            if change == 0:
                continue
            p2c[pair] += change
            if p2c[pair] > 0:
                heapq.heappush(heap, (-p2c[pair], ReverseSortPair(pair)))
        del p2c[best_pair]
        del p2w[best_pair]
    
    # 统计时间和内存
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    hours = (end_time - start_time) / 3600
    minutes = (end_time - start_time) / 60
    
    print(f"\n{'='*50}")
    print(f"Training Statistics:")
    print(f"{'='*50}")
    print(f"Time: {end_time - start_time:.2f} seconds ({minutes:.2f} minutes, {hours:.4f} hours)")
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"{'='*50}\n")
    
    return vocab, merges

def _validate_path(path: str | os.PathLike, path_name: str) -> Path:
    """验证路径是否有效，返回Path对象"""
    if path is None:
        raise ValueError(f"{path_name} cannot be None")
    
    path_obj = Path(path)
    
    # 检查路径是否为空
    if not str(path_obj).strip():
        raise ValueError(f"{path_name} cannot be empty")
    
    # 获取父目录
    parent_dir = path_obj.parent
    
    # 检查父目录是否存在，不存在则尝试创建
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create directory for {path_name}: {e}")
    
    # 检查父目录是否为目录
    if not parent_dir.is_dir():
        raise NotADirectoryError(f"Parent of {path_name} is not a directory: {parent_dir}")
    
    # 检查是否有写入权限
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"No write permission for directory of {path_name}: {parent_dir}")
    
    return path_obj


def save_bpe_results(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],vocab_path: str | os.PathLike, merges_path: str | os.PathLike):
    # 验证两个路径
    vocab_path_obj = _validate_path(vocab_path, "vocab_path")
    merges_path_obj = _validate_path(merges_path, "merges_path")
    
    # 人类可读的格式：bytes 转为 UTF-8 字符串
    # 使用 errors='replace' 确保所有内容都可读
    vocab_serialziable = {
        str(k): v.decode('utf-8', errors='replace')
          for k, v in vocab.items()
    }
    with open(vocab_path_obj, 'w', encoding='utf-8') as f:
          json.dump(vocab_serialziable, f, indent=2, ensure_ascii=False)

    merges_serializable = [
        [b1.decode('utf-8', errors='replace'), b2.decode('utf-8', errors='replace')]
        for b1, b2 in merges
    ]
    with open(merges_path_obj, 'w', encoding='utf-8') as f:
        json.dump(merges_serializable, f, indent=2, ensure_ascii=False)

    print(f"Vocab saved to {vocab_path_obj}, Merges saved to {merges_path_obj}")

def _get_byte_unicode_maps():
    """返回 bytes_to_unicode 和 unicode_to_bytes 映射字典"""
    bs = list(range(ord("!"), ord("~")+1)) + \
       list(range(ord("¡"), ord("¬")+1)) + \
       list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    bytes_to_unicode = dict(zip(bs, cs))      # {0: 'Ā', 32: 'Ġ', 195: 'Ã', ...}
    unicode_to_bytes = {v: k for k, v in bytes_to_unicode.items()}
    return bytes_to_unicode, unicode_to_bytes

def _unicode_to_bytes():
    """兼容旧代码，只返回 unicode_to_bytes 映射"""
    _, unicode_to_bytes = _get_byte_unicode_maps()
    return unicode_to_bytes

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes], merges:list[tuple[bytes,bytes]], special_tokens: list[str] | None = None):
        self.vocab_ = vocab
        self.bytes_to_id: dict[bytes, int] = {v:k for k, v in self.vocab_.items()}
        self.merges_ = merges
        self.merges_rank_ = {pair:idx for idx, pair in enumerate(self.merges_)}
        self.special_tokens_ = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.special_pattern_ = None
        self.eot_ids = {}
        self.bytes_to_unicode_, self.unicode_to_bytes_ = _get_byte_unicode_maps()
        if special_tokens:
            self.special_pattern_ = '|'.join(re.escape(token) for token in self.special_tokens_)
            for token in self.special_tokens_:
                token_bytes = token.encode("utf-8")
                if token_bytes in self.bytes_to_id:
                    self.eot_ids[token_bytes] = self.bytes_to_id[token_bytes]

    @classmethod
    def from_files(cls, vocab_filepath, merge_filepath, special_tokens = None):
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merge_filepath)
        if not vocab_path.exists or not merges_path.exists():
            raise FileExistsError       
        assert vocab_path.suffix == ".json", "Current vocab only supports json"
        assert merges_path.suffix == ".txt", "Current vocab only supports txt"
        # 阅读 vocab
        raw_vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        vocab: dict[int, bytes] = {}
        unicode_to_bytes = _unicode_to_bytes()
        for token_str, token_id in raw_vocab.items():
            token_bytes = bytes([unicode_to_bytes[c] for c in token_str])
            vocab[token_id] = token_bytes
        # 阅读 merges
        raw_merges = merges_path.read_text(encoding="utf-8")
        def token_str_to_bytes(token_str):
            return bytes([unicode_to_bytes[c] for c in token_str])
        merges: list[tuple[bytes,bytes]] = [
            (token_str_to_bytes(token1), token_str_to_bytes(token2)) for line in raw_merges.strip().split('\n') for token1,token2 in [line.split()]
        ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text:str) -> list[int]:
        chunks = [text]
        special_tokens = None
        if self.special_pattern_:
            chunks: list[str] = re.split(self.special_pattern_, text)
            special_tokens: list[str] = re.findall(self.special_pattern_, text)
            if len(special_tokens) == 0:
                special_tokens = None
        j = 0
        results:list[int] = []
        for chunk in chunks:
            tokens = re.findall(PAT, chunk)
            chunk_ids = []
            for token in tokens:
                if token:
                    # 1. 先将 token 编码为 UTF-8 bytes
                    # 2. 将每个 byte 映射为对应的 unicode 字符（如 byte 32 -> 'Ġ'）
                    # 3. 将这些 unicode 字符重新组合成字符串进行 BPE 合并
                    token_utf8_bytes = token.encode('utf-8')
                    token_unicode_chars = [self.bytes_to_unicode_[b] for b in token_utf8_bytes]
                    token_unicode_str = ''.join(token_unicode_chars)
                    # 现在 token_unicode_str 中的每个字符对应一个原始 byte
                    token_bytes: list[bytes] = [bytes([self.unicode_to_bytes_[c]]) for c in token_unicode_str]
                    while len(token_bytes) >= 2:
                        best_pair_idx = -1
                        min_merged_rank = float("inf")
                        merged = None
                        for i in range(len(token_bytes) - 1):
                            pair = (token_bytes[i], token_bytes[i + 1])
                            if pair in self.merges_ and self.merges_rank_[pair] < min_merged_rank:
                                best_pair_idx = i
                                merged = pair[0] + pair[1]
                                min_merged_rank = self.merges_rank_[pair]
                        if best_pair_idx == -1:
                            break
                        token_bytes = token_bytes[:best_pair_idx] + [merged] + token_bytes[best_pair_idx + 2:]
                    merged_id = [self.bytes_to_id[b] for b in token_bytes]
                    chunk_ids.extend(merged_id)

            if special_tokens and j < len(chunks) - 1:
                chunk_ids.append(self.eot_ids[special_tokens[j].encode("utf-8")])
            results.extend(chunk_ids)
            j += 1
        return results
        
    def encode_iterable(self, iterable:Iterable[str]) -> Iterator[int]:
        for i, text in enumerate(iterable):
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # 将 ids 转换为 bytes，然后解码为 UTF-8 字符串
        token_bytes = b''.join([self.vocab_[i] for i in ids])
        return token_bytes.decode('utf-8', errors='replace')

if __name__ == "__main__":
    vocab_file_path = "tests/fixtures/gpt2_vocab.json"
    merges_file_path = "tests/fixtures/gpt2_merges.txt"

    tokenizer = Tokenizer.from_files(vocab_file_path,merges_file_path,special_tokens=["<|endoftext|>"])
    text = ""
    encode_result = tokenizer.encode(text)
    decode_result = tokenizer.decode(encode_result)
    assert decode_result == text
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string
    # print(tokenizer.merges_[6])
