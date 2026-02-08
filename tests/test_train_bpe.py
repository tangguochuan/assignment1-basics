import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode
import pdb

import json

def save_bpe_results(vocab, merges, vocab_path, merges_path):
    """
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]
    """
    
    # --- 保存 Vocab 为 JSON ---
    # 由于 bytes 不能直接写入 JSON，我们需要把 bytes 转换成可读形式
    # 这里我们采用简单的映射：将 bytes 转换为 unicode 字符串以便查看
    # 或者如果你想更标准，可以使用之前提到的 gpt2_bytes_to_unicode 映射
    
    # 简单的转换方案：把 bytes 转化成 list of integers 或解码字符串
    serializable_vocab = {}
    for idx, b_slice in vocab.items():
        # 方案：保存为 ID: 字符串（如果解码失败则保留 repr）
        try:
            token_str = b_slice.decode('utf-8')
        except UnicodeDecodeError:
            token_str = b_slice.hex() # 或者用 repr(b_slice)
        serializable_vocab[idx] = token_str

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=4, ensure_ascii=False)

    # --- 保存 Merges 为 TXT ---
    # 格式通常为：token1 token2
    with open(merges_path, 'w', encoding='utf-8') as f:
        # 如果你想兼容 GPT-2 格式，需要把 bytes 转回 unicode 字符串
        # 这里为了简单，我们保存原始字节的 hex 或简单 decode
        for p1, p2 in merges:
            try:
                s1, s2 = p1.decode('utf-8'), p2.decode('utf-8')
                f.write(f"{s1} {s2}\n")
            except UnicodeDecodeError:
                f.write(f"{p1.hex()} {p2.hex()}\n")

    print(f"Vocab saved to {vocab_path}")
    print(f"Merges saved to {merges_path}")

def save_for_gpt2_test(vocab, merges, vocab_path, merges_path):
    # 获取 byte -> unicode 的映射表
    byte_to_unicode = gpt2_bytes_to_unicode()
    
    # 转换函数：将 bytes 对象转换为 GPT-2 风格的 unicode 字符串
    def b_to_u(b_obj):
        return "".join(byte_to_unicode[b] for b in b_obj)

    # 1. 保存 Merges (txt)
    with open(merges_path, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            f.write(f"{b_to_u(p1)} {b_to_u(p2)}\n")

    # 2. 保存 Vocab (json) 
    # 注意：参考文件里通常是 { "token_str": index }
    # 按 token id 排序，使其更易于人类阅读
    reference_style_vocab = {b_to_u(b_val): idx for idx, b_val in sorted(vocab.items())}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(reference_style_vocab, f, indent=2, ensure_ascii=False, sort_keys=False)

def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    # pdb.set_trace()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    my_vocab_json = FIXTURES_PATH / "my-vocab.json"
    my_merges_txt = FIXTURES_PATH / "my-merges.txt"
    save_for_gpt2_test(vocab, merges, my_vocab_json, my_merges_txt)
    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"
    
    # pdb.set_trace()
    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
