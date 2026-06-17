import tempfile
import unittest
from pathlib import Path

from src.common.env import load_env_file
from src.common.io import read_jsonl, write_jsonl


class CommonUtilityTests(unittest.TestCase):
    def test_load_env_file_parses_comments_quotes_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "sample.env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "MODEL_PATH=/models/pangu",
                        "WORK_ROOT=/work/openpangu_lab",
                        "CACHE_PATH=${WORK_ROOT}/cache/sft",
                        "export RUN_MODE='smoke'",
                        'OUTPUT_ROOT="outputs/exp2_1"',
                        "EMPTY=",
                    ]
                ),
                encoding="utf-8",
            )

            values = load_env_file(env_path)

        self.assertEqual(values["MODEL_PATH"], "/models/pangu")
        self.assertEqual(values["CACHE_PATH"], "/work/openpangu_lab/cache/sft")
        self.assertEqual(values["RUN_MODE"], "smoke")
        self.assertEqual(values["OUTPUT_ROOT"], "outputs/exp2_1")
        self.assertEqual(values["EMPTY"], "")

    def test_jsonl_roundtrip_preserves_unicode(self):
        rows = [{"id": 1, "text": "流浪地球"}, {"id": 2, "text": "copywriting"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, rows)
            loaded = list(read_jsonl(path))

        self.assertEqual(loaded, rows)


if __name__ == "__main__":
    unittest.main()
