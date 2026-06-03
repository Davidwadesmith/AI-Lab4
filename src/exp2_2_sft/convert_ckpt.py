from __future__ import annotations


def build_hf_to_mcore_command(
    *,
    code_root: str,
    load_hf_dir: str,
    save_mcore_dir: str,
    tokenizer_model: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int = 1,
    dtype: str = "bf16",
) -> str:
    return (
        f"cd {code_root} && python convert_ckpt.py "
        "--model-type GPT "
        "--load-model-type hf "
        "--save-model-type mg "
        f"--target-tensor-parallel-size {tensor_parallel_size} "
        f"--target-pipeline-parallel-size {pipeline_parallel_size} "
        f"--load-dir {load_hf_dir} "
        f"--save-dir {save_mcore_dir} "
        f"--tokenizer-model {tokenizer_model} "
        "--add-qkv-bias --add-dense-bias "
        f"--params-dtype {dtype} "
        "--use-mcore-models"
    )


def build_mcore_to_hf_command(
    *,
    code_root: str,
    load_mcore_dir: str,
    save_hf_dir: str,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    dtype: str = "bf16",
) -> str:
    return (
        f"cd {code_root} && python convert_ckpt.py "
        "--model-type GPT "
        "--load-model-type mg "
        "--save-model-type hf "
        f"--save-dir {save_hf_dir} "
        f"--load-dir {load_mcore_dir} "
        "--add-qkv-bias --add-dense-bias "
        f"--target-tensor-parallel-size {tensor_parallel_size} "
        f"--target-pipeline-parallel-size {pipeline_parallel_size} "
        f"--params-dtype {dtype} "
        "--use-mcore-models"
    )
