from __future__ import annotations

from typing import Iterable


def build_run_matrix(
    prefix: str,
    *,
    learning_rates: Iterable[str],
    global_batch_sizes: Iterable[int],
) -> list[dict[str, object]]:
    runs = []
    for learning_rate in learning_rates:
        for global_batch_size in global_batch_sizes:
            runs.append(
                {
                    "name": f"{prefix}_lr{learning_rate}_gbs{global_batch_size}",
                    "learning_rate": learning_rate,
                    "global_batch_size": int(global_batch_size),
                }
            )
    return runs


def _parent_dir(path: str) -> str:
    normalized = path.rstrip("/")
    if "/" in normalized:
        return normalized.rsplit("/", 1)[0]
    return "."


def build_train_command(
    *,
    code_root: str,
    npu_devices: str,
    npus_per_node: int,
    master_port: int,
    data_path: str,
    tokenizer_model: str,
    ckpt_load_dir: str,
    ckpt_save_dir: str,
    log_path: str,
    learning_rate: str,
    global_batch_size: int,
    train_iters: int,
    seq_length: int,
    save_interval: int = 100,
    lr_warmup_iters: int = 200,
    no_save_optim: bool = False,
    accumulate_allreduce_grads_in_fp32: bool = False,
    reuse_fp32_param: bool = True,
    overlap_param_gather: bool = False,
    overlap_grad_reduce: bool = False,
    use_distributed_optimizer: bool = False,
    npu_alloc_conf: str = "expandable_segments:True,max_split_size_mb:64",
) -> str:
    log_dir = _parent_dir(log_path)
    no_save_optim_arg = "  --no-save-optim \\\n" if no_save_optim else ""
    fp32_allreduce_arg = "  --accumulate-allreduce-grads-in-fp32 \\\n" if accumulate_allreduce_grads_in_fp32 else ""
    reuse_fp32_param_arg = "  --reuse-fp32-param \\\n" if reuse_fp32_param else ""
    overlap_param_gather_arg = "  --overlap-param-gather \\\n" if overlap_param_gather else ""
    overlap_grad_reduce_arg = "  --overlap-grad-reduce \\\n" if overlap_grad_reduce else ""
    distributed_optimizer_arg = "  --use-distributed-optimizer \\\n" if use_distributed_optimizer else ""
    command = f"""cd {code_root} && mkdir -p {log_dir} {ckpt_save_dir} && \\
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \\
export PYTORCH_NPU_ALLOC_CONF={npu_alloc_conf} && \\
export ASCEND_RT_VISIBLE_DEVICES="{npu_devices}" && \\
torchrun \\
  --nproc_per_node {npus_per_node} \\
  --nnodes 1 \\
  --node_rank 0 \\
  --master_addr localhost \\
  --master_port {master_port} \\
  posttrain_gpt.py \\
  --finetune \\
  --stage sft \\
  --use-mcore-models \\
  --tensor-model-parallel-size {npus_per_node} \\
  --pipeline-model-parallel-size 1 \\
  --context-parallel-size 1 \\
  --expert-model-parallel-size 1 \\
  --sequence-parallel \\
  --seed 1234 \\
  --num-layers 34 \\
  --hidden-size 4096 \\
  --ffn-hidden-size 12800 \\
  --num-attention-heads 32 \\
  --group-query-attention \\
  --num-query-groups 8 \\
  --kv-channels 128 \\
  --tokenizer-type PretrainedFromHF \\
  --tokenizer-not-use-fast \\
  --tokenizer-name-or-path {tokenizer_model} \\
  --seq-length {seq_length} \\
  --max-position-embeddings 32768 \\
  --micro-batch-size 1 \\
  --global-batch-size {global_batch_size} \\
  --transformer-impl local \\
  --distributed-timeout-minutes 120 \\
  --make-vocab-size-divisible-by 16 \\
  --padded-vocab-size 153376 \\
  --lr-decay-style cosine \\
  --lr {learning_rate} \\
  --min-lr 5e-7 \\
  --lr-warmup-iters {lr_warmup_iters} \\
  --override-opt_param-scheduler \\
  --disable-bias-linear \\
  --add-qkv-bias \\
  --add-dense-bias \\
  --attention-dropout 0.0 \\
  --init-method-std 0.01 \\
  --hidden-dropout 0.0 \\
  --position-embedding-type rope \\
  --use-rotary-position-embeddings \\
  --rotary-base 16000000 \\
  --normalization RMSNorm \\
  --swiglu \\
  --no-masked-softmax-fusion \\
  --no-gradient-accumulation-fusion \\
  --use-fused-rmsnorm \\
  --use-fused-swiglu \\
  --use-flash-attn \\
  --use-fused-rotary-pos-emb \\
  --attention-softmax-in-fp32 \\
{fp32_allreduce_arg}\
  --optimizer adam \\
  --weight-decay 0.1 \\
  --clip-grad 1.0 \\
  --loss-scale 1.0 \\
  --adam-beta1 0.9 \\
  --adam-beta2 0.95 \\
  --initial-loss-scale 4096 \\
  --no-load-optim \\
  --no-load-rng \\
  --bf16 \\
  --reset-attention-mask \\
  --reset-position-ids \\
  --eod-mask-loss \\
  --is-instruction-dataset \\
  --train-iters {train_iters} \\
  --recompute-granularity full \\
  --recompute-method block \\
  --recompute-num-layers 34 \\
{overlap_param_gather_arg}\
{overlap_grad_reduce_arg}\
{distributed_optimizer_arg}\
{reuse_fp32_param_arg}\
  --manual-gc \\
  --manual-gc-interval 100 \\
{no_save_optim_arg}\
  --untie-embeddings-and-output-weights \\
  --data-path {data_path} \\
  --split 100,0,0 \\
  --log-interval 1 \\
  --save-interval {save_interval} \\
  --eval-interval 100 \\
  --eval-iters 0 \\
  --distributed-backend nccl \\
  --load {ckpt_load_dir} \\
  --save {ckpt_save_dir} \\
  | tee {log_path}"""
    return f"bash -lc 'set -o pipefail; {command}'"
