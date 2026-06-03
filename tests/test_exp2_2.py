import tempfile
import unittest
import sys
from types import SimpleNamespace
from pathlib import Path

from src.common.io import read_jsonl
from src.exp2_2_sft.parse_logs import parse_training_log
from src.exp2_2_sft.prepare_data import convert_xhs_records, load_dataset_records, should_convert_mcore
from src.exp2_2_sft.train import build_run_matrix


class SftExperimentTests(unittest.TestCase):
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

    def test_load_dataset_records_reads_configured_split(self):
        original = sys.modules.get("datasets")
        sys.modules["datasets"] = SimpleNamespace(
            load_dataset=lambda path: {"train": [{"repo_name": "xhs/xhs", "instruction": "a", "output": "b"}]}
        )
        try:
            records = list(load_dataset_records("/dataset/path", split="train"))
        finally:
            if original is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = original

        self.assertEqual(records[0]["instruction"], "a")


if __name__ == "__main__":
    unittest.main()
