import os
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class BaselineQuantizationConfig(QuantizationConfig):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def get_name(cls) -> str:
        return "baseline"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "BaselineQuantizationConfig":
        return cls()

    # --- THE TRUE KILL-SWITCH ---
    # Placed on the Config class where vLLM actually checks it.
    @property
    def is_quantized(self) -> bool:
        return False

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            return BaselineLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return BaselineEmbeddingMethod(self)
        return None


def get_dynamic_watermark_bytes() -> int:
    budget_str = os.environ.get("RANS_WEIGHT_BUDGET_BYTES", str(6 * 1024**3))
    return int(budget_str)


class BaselineLinearMethod(LinearMethodBase):
    def __init__(self, quant_config=None):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **kwargs,
    ):
        target_device = torch.device(
            "cuda"
            if torch.cuda.memory_allocated() < get_dynamic_watermark_bytes()
            else "cpu"
        )

        # Ensure output_partition_sizes is a list for the logical_widths map
        if isinstance(output_partition_sizes, int):
            output_partition_sizes = [output_partition_sizes]

        out_features = sum(output_partition_sizes)

        weight = torch.empty(
            out_features,
            input_size_per_partition,
            dtype=params_dtype,
            device=target_device,
        )

        if target_device.type == "cpu":
            weight = weight.pin_memory()

        param = torch.nn.Parameter(weight, requires_grad=False)

        # 1. FORCE THE MAP: Give the layer the slice boundaries
        layer.logical_widths = output_partition_sizes

        # 2. ATTACH THE LOADER: Tell the param to use the layer's native slicing logic
        if hasattr(layer, "weight_loader"):
            param.weight_loader = layer.weight_loader

        layer.register_parameter("weight", param)

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        weight_on_gpu = layer.weight.to(x.device, non_blocking=True)
        return F.linear(x, weight_on_gpu, bias)


class BaselineEmbeddingMethod(LinearMethodBase):
    def __init__(self, quant_config=None):
        self.quant_config = quant_config

    def create_weights(self, layer, *args, **kwargs):
        target_device = torch.device(
            "cuda"
            if torch.cuda.memory_allocated() < get_dynamic_watermark_bytes()
            else "cpu"
        )

        num_embeddings = getattr(
            layer,
            "num_embeddings_per_partition",
            getattr(layer, "num_embeddings", 151936),
        )
        embedding_dim = getattr(layer, "embedding_dim", 5120)

        params_dtype = kwargs.get("params_dtype", torch.bfloat16)
        if params_dtype is None and len(args) > 4:
            params_dtype = args[4]

        weight = torch.empty(
            num_embeddings, embedding_dim, dtype=params_dtype, device=target_device
        )

        if target_device.type == "cpu":
            weight = weight.pin_memory()

        param = torch.nn.Parameter(weight, requires_grad=False)

        # Attach the native loader for embeddings (handles vocab padding)
        if hasattr(layer, "weight_loader"):
            param.weight_loader = layer.weight_loader

        layer.register_parameter("weight", param)

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        weight_on_gpu = layer.weight.to(x.device, non_blocking=True)
        return F.embedding(x, weight_on_gpu)

    def embedding(self, layer, x, bias=None) -> torch.Tensor:
        weight_on_gpu = layer.weight.to(x.device, non_blocking=True)
        return F.embedding(x, weight_on_gpu)
