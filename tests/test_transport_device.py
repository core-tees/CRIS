"""Device selection and run telemetry without optimizer iterations."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/transport"))
import train_impl as trainer


class TransportDeviceTest(unittest.TestCase):
    def test_selected_device_controls_cuda_helpers_and_metadata(self):
        original_dtype = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original_dtype)
        torch.set_default_dtype(torch.float64)
        trainer.configure()
        as_tensor = torch.as_tensor
        initial = trainer.core.initial

        for selection in ("cpu", "cuda", "cuda:0", "cuda:1"):
            with self.subTest(device=selection), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root / "data"
                data.mkdir()
                (data / "READY.json").write_text("{}")
                for name in ("train", "validation", "endpoint", "quadratic", "singular"):
                    np.savez(
                        data / f"{name}.npz",
                        x=np.zeros((2, 3)),
                        y=np.array([0.25, 0.75]),
                        endpoint=np.array([False, True]),
                    )
                args = argparse.Namespace(
                    device=selection, out=root / "run", data=data,
                    seed=42, active=2, method="lm", growth=False, seconds=0,
                )
                device = torch.device(selection)
                # Keep the numerical data on CPU so indexed-device checks also
                # run on machines without CUDA or with only one GPU.
                with ExitStack() as stack:
                    stack.enter_context(patch.object(
                        torch, "as_tensor",
                        side_effect=lambda value, **kwargs: as_tensor(value, device="cpu"),
                    ))
                    initializer = stack.enter_context(patch.object(
                        trainer.core, "initial",
                        side_effect=lambda seed, device: initial(seed, "cpu"),
                    ))
                    preflight = stack.enter_context(patch.object(
                        trainer, "preflight", return_value={},
                    ))
                    cuda = {
                        name: stack.enter_context(patch.object(torch.cuda, name, return_value=value))
                        for name, value in (
                            ("synchronize", None), ("get_device_name", "selected GPU"),
                            ("reset_peak_memory_stats", None), ("max_memory_allocated", 1234),
                        )
                    }
                    trainer.run(args)
                preflight.assert_called_once_with(device)
                initializer.assert_called_once_with(42, device)
                identity = json.loads((args.out / "IDENTITY.json").read_text())
                complete = json.loads((args.out / "COMPLETE.json").read_text())
                self.assertEqual(complete["iteration"], 0)
                if device.type == "cuda":
                    for helper in cuda.values():
                        helper.assert_called_once_with(device)
                    self.assertEqual(identity["gpu"], "selected GPU")
                    self.assertEqual(complete["peak_gpu_memory_bytes"], 1234)
                else:
                    for helper in cuda.values():
                        helper.assert_not_called()
                    self.assertEqual(identity["gpu"], "cpu")
                    self.assertIsNone(complete["peak_gpu_memory_bytes"])


if __name__ == "__main__":
    unittest.main()
