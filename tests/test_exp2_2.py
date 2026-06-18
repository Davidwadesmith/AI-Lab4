import tempfile
import unittest
import sys
from types import SimpleNamespace
from pathlib import Path

from src.common.io import read_jsonl
from src.exp2_2_sft.main import _ensure_preprocess_output_dir, _join_config_path
from src.exp2_2_sft.parse_logs import parse_training_log
from src.exp2_2_sft.prepare_data import convert_xhs_records, load_dataset_records, should_convert_mcore
from src.exp2_2_sft.train import build_run_matrix, build_train_command


class SftExperimentTests(unittest.TestCase):
    def test_join_config_path_preserves_linux_absolute_paths(self):
        self.assertEqual(
            _join_config_path("/home/ma-user/work/openpangu_lab/outputs/exp2_2", "logs", "run.log"),
            "/home/ma-user/work/openpangu_lab/outputs/exp2_2/logs/run.log",
        )

    def test_convert_xhs_records_filters_and_formats_mindspeed_jsonl(self):
        records = [
            {"repo_name": "xhs/xhs", "instruction": "写一段小红书文案", "output": "好用到想分享"},
            {"repo_name": "other", "instruction": "skip", "output": "skip"},
        ]

        converted = list(convert_xhs_records(records))

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["meta_prompt"], [])
        self.assertEqual(converted[0]["data"][0]["role"], "user")
        self.assertEqual(converted[0]["data"][1]["role"], "assistant")

    def test_convert_xhs_records_respects_max_records(self):
        records = [
            {"repo_name": "xhs/xhs", "instruction": "a", "output": "b"},
            {"repo_name": "xhs/xhs", "instruction": "c", "output": "d"},
        ]

        converted = list(convert_xhs_records(records, max_records=1))

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["data"][0]["content"], "a")

    def test_parse_training_log_extracts_loss_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.log"
            path.write_text(
                "step 1 | lm loss: 3.2500 | tokens/s: 123.4\n"
                "step 2 | lm loss: 2.7500 | samples/s: 0.5\n",
                encoding="utf-8",
            )

            metrics = parse_training_log(path)

        self.assertEqual([row["step"] for row in metrics], [1, 2])
        self.assertEqual(metrics[0]["lm_loss"], 3.25)
        self.assertEqual(metrics[0]["tokens_per_second"], 123.4)
        self.assertEqual(metrics[1]["samples_per_second"], 0.5)

    def test_build_run_matrix_uses_isolated_names(self):
        matrix = build_run_matrix("smoke", learning_rates=["1e-5", "5e-6"], global_batch_sizes=[1])

        self.assertEqual([run["name"] for run in matrix], ["smoke_lr1e-5_gbs1", "smoke_lr5e-6_gbs1"])
        self.assertEqual(matrix[0]["learning_rate"], "1e-5")
        self.assertEqual(matrix[0]["global_batch_size"], 1)

    def test_should_convert_mcore_when_forced_or_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            missing = Path(tmp) / "missing"

            self.assertFalse(should_convert_mcore(existing, force=False))
            self.assertTrue(should_convert_mcore(existing, force=True))
            self.assertTrue(should_convert_mcore(missing, force=False))

    def test_ensure_preprocess_output_dir_creates_prefix_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "cache" / "finetune" / "sft"

            created = _ensure_preprocess_output_dir(str(prefix))

            self.assertTrue(created.exists())
            self.assertEqual(created, prefix.parent)

    def test_load_dataset_records_reads_configured_split(self):
        original = sys.modules.get("datasets")
        calls = []
        sys.modules["datasets"] = SimpleNamespace(
            load_dataset=lambda path, **kwargs: calls.append((path, kwargs))
            or {"train": [{"repo_name": "xhs/xhs", "instruction": "a", "output": "b"}]}
        )
        try:
            records = list(load_dataset_records("/dataset/path", split="train"))
        finally:
            if original is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = original

        self.assertEqual(records[0]["instruction"], "a")
        self.assertEqual(calls[0], ("/dataset/path", {"split": None, "streaming": False}))

    def test_load_dataset_records_supports_streaming_split(self):
        original = sys.modules.get("datasets")
        calls = []
        sys.modules["datasets"] = SimpleNamespace(
            load_dataset=lambda path, **kwargs: calls.append((path, kwargs))
            or iter([{"repo_name": "xhs/xhs", "instruction": "a", "output": "b"}])
        )
        try:
            records = list(load_dataset_records("/dataset/path", split="train", streaming=True))
        finally:
            if original is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = original

        self.assertEqual(records[0]["instruction"], "a")
        self.assertEqual(calls[0], ("/dataset/path", {"split": "train", "streaming": True}))

    def test_build_train_command_uses_space_saving_training_options(self):
        command = build_train_command(
            code_root="/home/ma-user/work/openpangu_lab/MindSpeed-LLM",
            npu_devices="0,1,2,3",
            npus_per_node=4,
            master_port=6000,
            data_path="/home/ma-user/work/openpangu_lab/cache/sft",
            tokenizer_model="/home/ma-user/work/openpangu_lab/models/openPangu-Embedded-7B-V1.1",
            ckpt_load_dir="/home/ma-user/work/openpangu_lab/ckpts/openPangu_7B_mcore",
            ckpt_save_dir="/home/ma-user/work/openpangu_lab/outputs/checkpoints/run",
            log_path="/home/ma-user/work/openpangu_lab/outputs/logs/run.log",
            learning_rate="5e-6",
            global_batch_size=4,
            train_iters=100,
            seq_length=8192,
            save_interval=100,
            lr_warmup_iters=20,
            no_save_optim=True,
        )

        self.assertIn("--global-batch-size 4", command)
        self.assertIn("--train-iters 100", command)
        self.assertIn("--save-interval 100", command)
        self.assertIn("--lr-warmup-iters 20", command)
        self.assertIn("--no-save-optim", command)
        self.assertIn("set -o pipefail", command)
        self.assertIn("PYTORCH_NPU_ALLOC_CONF=expandable_segments:True", command)
        self.assertIn("--reuse-fp32-param", command)
        self.assertNotIn("--accumulate-allreduce-grads-in-fp32", command)
        self.assertIn("mkdir -p /home/ma-user/work/openpangu_lab/outputs/logs /home/ma-user/work/openpangu_lab/outputs/checkpoints/run", command)

    def test_build_train_command_can_enable_fp32_grad_allreduce(self):
        command = build_train_command(
            code_root="/work/MindSpeed-LLM",
            npu_devices="0,1,2,3",
            npus_per_node=4,
            master_port=6000,
            data_path="/work/cache/sft",
            tokenizer_model="/work/model",
            ckpt_load_dir="/work/ckpts/mcore",
            ckpt_save_dir="/work/outputs/checkpoints/run",
            log_path="/work/outputs/logs/run.log",
            learning_rate="5e-5",
            global_batch_size=4,
            train_iters=100,
            seq_length=8192,
            accumulate_allreduce_grads_in_fp32=True,
        )

        self.assertIn("--accumulate-allreduce-grads-in-fp32", command)

    def test_build_train_command_can_disable_reuse_fp32_param(self):
        command = build_train_command(
            code_root="/work/MindSpeed-LLM",
            npu_devices="0,1,2,3",
            npus_per_node=4,
            master_port=6000,
            data_path="/work/cache/sft",
            tokenizer_model="/work/model",
            ckpt_load_dir="/work/ckpts/mcore",
            ckpt_save_dir="/work/outputs/checkpoints/run",
            log_path="/work/outputs/logs/run.log",
            learning_rate="5e-5",
            global_batch_size=4,
            train_iters=100,
            seq_length=8192,
            reuse_fp32_param=False,
        )

        self.assertNotIn("--reuse-fp32-param", command)


if __name__ == "__main__":
    unittest.main()
