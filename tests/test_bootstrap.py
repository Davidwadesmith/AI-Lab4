import unittest

from src.common.bootstrap import build_bootstrap_plan


class BootstrapPlanTests(unittest.TestCase):
    def test_exp2_1_plan_creates_venv_sources_ascend_and_downloads_model(self):
        plan = build_bootstrap_plan(
            {
                "AUTO_SETUP_ENV": "1",
                "PYTHON_BIN": "python3.10",
                "VENV_DIR": ".venv",
                "ASCEND_TOOLKIT_ENV": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
                "TORCH_INSTALL_COMMAND": "pip install torch==2.1.0",
                "TORCH_NPU_INSTALL_COMMAND": "pip install torch-npu==2.1.0",
                "MODEL_PATH": "/models/pangu",
                "MODEL_DOWNLOAD_COMMAND": "git clone https://example.invalid/pangu /models/pangu",
            },
            experiment="exp2_1",
        )

        self.assertIn("source /usr/local/Ascend/ascend-toolkit/set_env.sh", plan.commands)
        self.assertIn('if [[ ! -d ".venv" ]]; then python3.10 -m venv ".venv"; fi', plan.commands)
        self.assertIn('source ".venv/bin/activate"', plan.commands)
        self.assertIn("pip install torch==2.1.0", plan.commands)
        self.assertIn("pip install torch-npu==2.1.0", plan.commands)
        self.assertIn('if [[ ! -e "/models/pangu" ]]; then git clone https://example.invalid/pangu /models/pangu; fi', plan.commands)
        self.assertIn("python -m pip install --upgrade pip wheel", plan.commands)
        self.assertNotIn("python -m pip install --upgrade pip setuptools wheel", plan.commands)

    def test_setuptools_install_command_is_explicit(self):
        plan = build_bootstrap_plan(
            {
                "AUTO_SETUP_ENV": "1",
                "VENV_DIR": ".venv",
                "SETUPTOOLS_INSTALL_COMMAND": "python -m pip install setuptools==69.5.1",
            },
            experiment="exp2_2",
        )

        self.assertIn("python -m pip install setuptools==69.5.1", plan.commands)

    def test_venv_can_reuse_system_site_packages(self):
        plan = build_bootstrap_plan(
            {
                "AUTO_SETUP_ENV": "1",
                "PYTHON_BIN": "python3.10",
                "VENV_DIR": "/work/.venv",
                "VENV_SYSTEM_SITE_PACKAGES": "1",
            },
            experiment="exp2_2",
        )

        self.assertIn('if [[ ! -d "/work/.venv" ]]; then python3.10 -m venv --system-site-packages "/work/.venv"; fi', plan.commands)

    def test_ninja_install_command_is_guarded(self):
        plan = build_bootstrap_plan(
            {
                "AUTO_SETUP_ENV": "1",
                "VENV_DIR": ".venv",
                "NINJA_INSTALL_COMMAND": "python -m pip install ninja",
            },
            experiment="exp2_2",
        )

        self.assertIn(
            "if ! command -v ninja >/dev/null 2>&1; then python -m pip install ninja; fi",
            plan.commands,
        )

    def test_model_download_can_use_file_guard(self):
        plan = build_bootstrap_plan(
            {
                "AUTO_SETUP_ENV": "1",
                "VENV_DIR": ".venv",
                "HF_MODEL_PATH": "/models/pangu_hf",
                "MODEL_DOWNLOAD_GUARD": "/models/pangu_hf/config.json",
                "MODEL_DOWNLOAD_COMMAND": "python -m src.common.download_hf repo /models/pangu_hf",
            },
            experiment="exp2_2",
        )

        self.assertIn(
            'if [[ ! -e "/models/pangu_hf/config.json" ]]; then python -m src.common.download_hf repo /models/pangu_hf; fi',
            plan.commands,
        )

    def test_exp2_2_plan_clones_mindspeed_and_downloads_data(self):
        plan = build_bootstrap_plan(
            {
                "AUTO_SETUP_ENV": "1",
                "AUTO_CLONE_MINDSPEED": "1",
                "MINDSPEED_LLM_ROOT": "/work/MindSpeed-LLM",
                "MINDSPEED_LLM_REPO": "https://gitee.com/ascend/MindSpeed-LLM.git",
                "MINDSPEED_LLM_REF": "2.1.0",
                "DATA_DOWNLOAD_COMMAND": "python scripts/download_data.py",
                "SOURCE_JSONL": "/data/train.jsonl",
                "HF_MODEL_PATH": "/models/pangu_hf",
                "MODEL_DOWNLOAD_COMMAND": "git clone https://example.invalid/pangu /models/pangu_hf",
            },
            experiment="exp2_2",
        )

        self.assertIn(
            'if [[ ! -d "/work/MindSpeed-LLM/.git" ]]; then git clone "https://gitee.com/ascend/MindSpeed-LLM.git" "/work/MindSpeed-LLM"; fi',
            plan.commands,
        )
        self.assertIn('git -C "/work/MindSpeed-LLM" checkout "2.1.0"', plan.commands)
        self.assertIn('if [[ -f "/work/MindSpeed-LLM/.gitmodules" ]]; then git -C "/work/MindSpeed-LLM" submodule update --init --recursive; fi', plan.commands)
        self.assertIn('if [[ ! -e "/models/pangu_hf" ]]; then git clone https://example.invalid/pangu /models/pangu_hf; fi', plan.commands)
        self.assertIn('if [[ ! -e "/data/train.jsonl" ]]; then python scripts/download_data.py; fi', plan.commands)

    def test_auto_setup_disabled_returns_no_commands(self):
        plan = build_bootstrap_plan({"AUTO_SETUP_ENV": "0"}, experiment="exp2_1")
        self.assertEqual(plan.commands, [])


if __name__ == "__main__":
    unittest.main()
