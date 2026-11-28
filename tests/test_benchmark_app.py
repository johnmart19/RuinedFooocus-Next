"""The benchmark tool's parsing and summaries, without starting the app or a GPU.

These checks do not run a benchmark; they cover how counter rows, model ids and
result tables are read and reported.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import benchmark_app as bench  # noqa: E402


class BenchmarkToolTests(unittest.TestCase):
    def test_counter_rows_give_memory_and_busiest_engine_and_drop_garbage(self):
        header = bench.parse_csv_line(
            '"(PDH-CSV 4.0)","\\\\PC\\GPU Process Memory(pid_1_luid_0x0_phys_0)\\Dedicated Usage",'
            '"\\\\PC\\GPU Engine(pid_1_eng_0_engtype_3D)\\Utilization Percentage",'
            '"\\\\PC\\GPU Engine(pid_1_eng_1_engtype_Compute)\\Utilization Percentage"'
        )
        row = bench.parse_csv_line('"11/20/2026 21:54:04.680","2147483648.0","12.5","81.0"')
        sample = bench.parse_sample(header, row)
        self.assertEqual(sample["memory"], 2147483648.0)
        self.assertEqual(sample["load"], 81.0)
        # A counter that has just appeared can report a nonsense first value.
        garbage = bench.parse_csv_line(
            '"11/20/2026 21:54:05.680","2147483648.0","6761670822980.5"," "'
        )
        self.assertEqual(bench.parse_sample(header, garbage)["load"], 0.0)
        self.assertIsNone(bench.parse_csv_line("The command completed successfully."))

    def test_model_names_resolve_by_part_and_unavailable_models_are_skipped(self):
        models = {
            "data": [
                {"id": "chat/llm.gguf", "type": "chat"},
                {"id": "image/broken.safetensors", "type": "image", "generation_available": False},
                {"id": "image/animaMayhem_29BV10_fp16.safetensors", "type": "image"},
            ]
        }
        self.assertEqual(
            bench.resolve_model(models, "ANIMA"), "image/animaMayhem_29BV10_fp16.safetensors"
        )
        with self.assertRaises(SystemExit):
            bench.resolve_model(models, "broken")

    def test_tables_separate_the_first_image_from_later_ones(self):
        records = [
            {
                "model": "image/m",
                "size": "1024x1024",
                "run": 0,
                "seconds": 40.0,
                "gpu_memory_gb": 8.0,
            },
            {
                "model": "image/m",
                "size": "1024x1024",
                "run": 1,
                "seconds": 36.0,
                "gpu_busy_percent": 80.0,
            },
            {
                "model": "image/m",
                "size": "1024x1024",
                "run": 2,
                "seconds": 38.0,
                "gpu_busy_percent": 90.0,
            },
            {"model": "image/x", "size": "1024x1024", "run": 0, "error": "HTTP 500"},
        ]
        table = bench.generation_table(records)
        self.assertIn("| image/m | 1024x1024 | 40.0 | 37.0 (36.0-38.0) | 8.0 | 85.0 |", table)
        self.assertIn("| image/x | 1024x1024 | failed | - | - | - |", table)
        startup = bench.startup_table([{"page": 20.0, "config": 21.0}, {"page": 22.0}])
        self.assertIn("| page | 21.0 | 20.0 | 22.0 |", startup)
        self.assertIn("| config | 21.0 | 21.0 | 21.0 |", startup)


if __name__ == "__main__":
    unittest.main()
