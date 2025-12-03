import os
import torch

def setup_environment():
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["VLLM_USE_V1"] = "0"  # Disable v1 API for now since it does not support logits processors.
    os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.75"
    os.environ["VLLM_MAX_MODEL_LEN"] = "65536"
    # os.environ["VLLM_QUANTIZATION"] = None
    os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = "1"
    os.environ["VLLM_DTYPE"] = "float16"
    os.environ["VLLM_ENABLE_CHUNKED_PREFILL"] = "true"
    os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"] = "32768"
    os.environ["VLLM_MAX_NUM_SEQS"] = "16"
    os.environ["VLLM_BLOCK_SIZE"] = "8"
    os.environ["VLLM_SWAP_SPACE"] = "8"
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    # Key settings
    os.environ["USE_FLASH_ATTENTION"] = "2"
    os.environ["VLLM_ATTENTION_BACKEND"] = "flash-attn"  # Force use of Flash Attention
    os.environ["FLASH_ATTENTION_FORCE_BUILD"] = "1"  # Ensure recompilation

    # Environment check
    print("===== Environment Check =====")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    print("\n===== Environment Variables =====")
    print(f"USE_FLASH_ATTENTION={os.getenv('USE_FLASH_ATTENTION')}")
    print(f"VLLM_ATTENTION_BACKEND={os.getenv('VLLM_ATTENTION_BACKEND')}")

    # Check FlashAttention availability
    try:
        from vllm._C import ops
        if ops.is_flash_attn_available():
            print(f"✅ Kernel Check: FlashAttention-2 Enabled (Version: {ops.get_flash_attn_version()})")
        else:
            print("❌ Kernel Check: FlashAttention-2 Not Available")
    except ImportError:
        print("⚠️ Unable to import vLLM internal module")


setup_environment()