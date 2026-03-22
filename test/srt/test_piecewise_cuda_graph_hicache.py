"""
Test that hierarchical cache does not incorrectly disable piecewise CUDA graph.

FP8 flashinfer decode attention relies on piecewise CUDA graph (PCG) for correct
execution. Previously, `enable_hierarchical_cache` was incorrectly grouped with
`cpu_offload_gb` in the PCG disable logic, causing FP8 + hicache to produce
garbled output.

This test verifies the fix: hicache should NOT disable PCG, while cpu_offload
should still correctly disable PCG.
"""

import unittest

from unittest.mock import MagicMock, patch


class TestPiecewiseCudaGraphHicache(unittest.TestCase):
    """Test _handle_piecewise_cuda_graph does not disable PCG for hicache."""

    def _create_server_args(self, **overrides):
        """Create a minimal ServerArgs-like object for testing."""
        defaults = {
            "enable_hierarchical_cache": False,
            "cpu_offload_gb": 0,
            "speculative_algorithm": None,
            "enable_dp_attention": False,
            "enable_torch_compile": False,
            "pp_size": 1,
            "moe_a2a_backend": "none",
            "lora_paths": None,
            "enable_lora": None,
            "load_format": "auto",
            "quantization": None,
            "model_path": "dummy",
            "dllm_algorithm": None,
            "enable_deterministic_inference": False,
            "disaggregation_mode": "null",
            "enable_symm_mem": False,
            "enable_eplb": False,
            "expert_distribution_recorder_mode": None,
            "disable_piecewise_cuda_graph": False,
            "enforce_piecewise_cuda_graph": False,
            "piecewise_cuda_graph_compiler": "eager",
        }
        defaults.update(overrides)

        args = MagicMock()
        for k, v in defaults.items():
            setattr(args, k, v)

        # Mock model config
        model_config = MagicMock()
        model_config.is_piecewise_cuda_graph_disabled_model = False
        model_config.is_multimodal = False
        args.get_model_config.return_value = model_config

        return args

    def test_hicache_does_not_disable_pcg(self):
        """Hierarchical cache should NOT disable piecewise CUDA graph."""
        from sglang.srt.server_args import ServerArgs

        args = self._create_server_args(enable_hierarchical_cache=True)
        ServerArgs._handle_piecewise_cuda_graph(args)

        self.assertFalse(
            args.disable_piecewise_cuda_graph,
            "enable_hierarchical_cache should NOT disable piecewise CUDA graph. "
            "HiCache operates at scheduler level (KV eviction/restore between forwards) "
            "and does not affect CUDA graph recording or replay.",
        )

    def test_cpu_offload_still_disables_pcg(self):
        """CPU offload should still correctly disable piecewise CUDA graph."""
        from sglang.srt.server_args import ServerArgs

        args = self._create_server_args(cpu_offload_gb=4)
        ServerArgs._handle_piecewise_cuda_graph(args)

        self.assertTrue(
            args.disable_piecewise_cuda_graph,
            "cpu_offload_gb > 0 should disable piecewise CUDA graph. "
            "CPU offload moves model weights to CPU during forward, "
            "changing GPU tensor addresses and breaking CUDA graph replay.",
        )

    def test_hicache_with_cpu_offload_disables_pcg(self):
        """When both hicache and cpu_offload are enabled, PCG should be disabled (by cpu_offload)."""
        from sglang.srt.server_args import ServerArgs

        args = self._create_server_args(
            enable_hierarchical_cache=True, cpu_offload_gb=4
        )
        ServerArgs._handle_piecewise_cuda_graph(args)

        self.assertTrue(
            args.disable_piecewise_cuda_graph,
            "cpu_offload_gb > 0 should disable PCG even when hicache is also enabled.",
        )

    def test_no_hicache_no_offload_keeps_pcg(self):
        """Without hicache or cpu_offload, PCG should remain enabled."""
        from sglang.srt.server_args import ServerArgs

        args = self._create_server_args()
        ServerArgs._handle_piecewise_cuda_graph(args)

        self.assertFalse(
            args.disable_piecewise_cuda_graph,
            "PCG should remain enabled when no disabling conditions are met.",
        )


if __name__ == "__main__":
    unittest.main()
