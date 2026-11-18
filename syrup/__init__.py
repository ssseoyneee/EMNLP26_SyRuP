"""SyRuP: system-prompt following via reward-guided prediction in LLM decoding."""
from .data import encode_record
from .generate import syrup_generate
from .modeling import RewardHead, SyRuPModel, load_checkpoint, load_tokenizer

__all__ = ["RewardHead", "SyRuPModel", "encode_record", "load_checkpoint", "load_tokenizer", "syrup_generate"]
