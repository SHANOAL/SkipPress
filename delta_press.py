# delta_press.py
import logging
from dataclasses import dataclass,field
from typing import Optional, Dict, Tuple, List, Literal
import torch
from torch import nn
from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)




@dataclass
class DeltaPressViT:
  

    compression_ratio: float = 0.5
    start_ratio: float = 0.3
    end_ratio: float = 1.0
    metric: Literal["cos", "l1", "l2"] = "l2"
    decide_once_in_flexible: bool = True
    stability_threshold: float = 0.0

    decide_interval: int = 0

    protected_token_indices: Tuple[int, ...] = (0,)

    _global_active_mask: Optional[torch.Tensor] = None
    _active_pairs: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    _vit_total_layers: Optional[int] = None

    def __post_init__(self):
        self._active_pairs = {}

    def _flexible_range(self, module: nn.Module) -> Tuple[int, int]:
        total_layers = self._vit_total_layers
        if total_layers is None:
            cfg = getattr(module, "config", None)
            total_layers = getattr(cfg, "num_hidden_layers", None) \
                or getattr(cfg, "num_layers", None) \
                or getattr(cfg, "depth", None) \
                or 32
        start_layer = int(total_layers * self.start_ratio)
        end_layer = int(total_layers * self.end_ratio)
        end_layer = max(start_layer + 1, end_layer)
        return start_layer, end_layer

    @staticmethod
    def _dash_score_delta(attn_out: torch.Tensor, attn_in: torch.Tensor) -> torch.Tensor:
        delta = (attn_out - attn_in).norm(dim=-1)  # (B,N) or (N,)
        if delta.dim() == 2:
            return delta.mean(dim=0)               # (N,)
        return delta                               # (N,)

    @staticmethod
    def _is_lengths_vec(x: torch.Tensor, ref_len: int) -> bool:
        # lengths: 1D int, sum == ref_len, all > 0
        if x.dim() != 1 or x.numel() < 1:
            return False
        if not (x.dtype in (torch.int32, torch.int64, torch.long)):
            return False
        s = int(x.sum().item())
        if s != ref_len:
            return False
        if torch.any(x <= 0):
            return False
        return True

    @staticmethod
    def _is_cu_seqlens(x: torch.Tensor, ref_len: int) -> bool:
        # cu_seqlens: 1D int, starts with 0, ends with ref_len, nondecreasing
        if x.dim() != 1 or x.numel() < 2:
            return False
        if not (x.dtype in (torch.int32, torch.int64, torch.long)):
            return False
        if int(x[0].item()) != 0:
            return False
        if int(x[-1].item()) != ref_len:
            return False
        if torch.any(x[1:] < x[:-1]):
            return False
        return True

    @staticmethod
    def _ranges_from_lengths(lengths: torch.Tensor) -> List[Tuple[int, int]]:
        # returns list of (start, end) over packed seq positions
        cumsum = torch.cumsum(lengths, dim=0).tolist()
        starts = [0] + cumsum[:-1]
        return list(zip(starts, cumsum))

    def _enforce_keep_one_per_segment(self, mask: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Ensure each segment keeps at least 1 token, otherwise torch.split with that segment dies.
        We do NOT change lengths here, only fix mask.
        """
        if lengths is None or (not torch.is_tensor(lengths)) or (not self._is_lengths_vec(lengths, mask.numel())):
            return mask

        fixed = mask.clone()
        for (s, e) in self._ranges_from_lengths(lengths):
            if s >= e:
                continue
            if int(fixed[s:e].sum().item()) == 0:
                fixed[s] = True  # keep the first token in that segment
        return fixed

    def _update_lengths_by_mask(self, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Given original lengths summing to ref_len, compute new lengths after pruning by mask.
        Also enforces min 1 per segment (by clamping to 1, assuming mask already enforced).
        """
        new_lens = []
        for (s, e) in self._ranges_from_lengths(lengths):
            kept = int(mask[s:e].sum().item())
            if kept <= 0:
                kept = 1
            new_lens.append(kept)
        return torch.tensor(new_lens, device=lengths.device, dtype=lengths.dtype)

    def _update_cu_seqlens_by_mask(self, cu: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        cu_seqlens -> lengths -> new_lengths -> new_cu_seqlens
        """
        lengths = cu[1:] - cu[:-1]
        # lengths sum == ref_len by definition
        new_lengths = self._update_lengths_by_mask(lengths, mask)
        new_cu = torch.zeros((new_lengths.numel() + 1,), device=cu.device, dtype=cu.dtype)
        new_cu[1:] = torch.cumsum(new_lengths, dim=0)
        return new_cu

    def _recursive_prune(self, data: any, mask: torch.Tensor, ref_len: int, name_hint: str = "") -> any:
        """
        Brutal recursive pruning:
        - Any tensor whose seq dimension matches ref_len will be pruned.
        - Special handling for `lengths` and `cu_seqlens` so split() won't crash.
        """
        if data is None:
            return None

        if isinstance(data, tuple):
            return tuple(self._recursive_prune(item, mask, ref_len, f"{name_hint}[{i}]")
                         for i, item in enumerate(data))

        if isinstance(data, list):
            return [self._recursive_prune(item, mask, ref_len, f"{name_hint}[{i}]")
                    for i, item in enumerate(data)]

        if isinstance(data, dict):
            return {k: self._recursive_prune(v, mask, ref_len, f"{name_hint}.{k}")
                    for k, v in data.items()}

        if torch.is_tensor(data):
            # ---- Special: packed segment lengths ----
            if self._is_lengths_vec(data, ref_len):
                # update lengths to match pruned seq
                return self._update_lengths_by_mask(data, mask)

            # ---- Special: cu_seqlens ----
            if self._is_cu_seqlens(data, ref_len):
                return self._update_cu_seqlens_by_mask(data, mask)

            if data.shape[0] == ref_len:
                return data[mask]

            if data.dim() >= 2 and data.shape[1] == ref_len:
                return data[:, mask]

            if data.dim() >= 3 and data.shape[-1] == ref_len:
                try:
                    if data.shape[-2] == ref_len:
                        return data[..., mask, :][..., :, mask]
                    else:
                        return data[..., mask]
                except Exception:
                    pass

        return data

    # ---------------- Hooks ----------------
    def forward_pre_hook(self, module: nn.Module, args, kwargs):
        # 1) find hidden_states
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]
        if hidden_states is None or (not torch.is_tensor(hidden_states)):
            return args, kwargs

        # seq_len: Qwen2VL vision typically (B, N, D) or sometimes (N, D)
        seq_len = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]

        layer_idx = getattr(module, "layer_idx", 0)
        start_layer, end_layer = self._flexible_range(module)
        in_flexible = start_layer <= layer_idx < end_layer

        active_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)

        if in_flexible and self._global_active_mask is not None and self._global_active_mask.shape[0] == seq_len:
            active_mask = self._global_active_mask.to(hidden_states.device)

        # protect CLS/global indices
        for t in self.protected_token_indices:
            if 0 <= t < seq_len:
                active_mask[t] = True

        # IMPORTANT: enforce keep-one per segment if lengths exists in kwargs
        # (we search for a lengths vector that sums to seq_len)
        lengths_in_kwargs = None
        for v in kwargs.values():
            if torch.is_tensor(v) and self._is_lengths_vec(v, seq_len):
                lengths_in_kwargs = v
                break
            if torch.is_tensor(v) and self._is_cu_seqlens(v, seq_len):
                lengths_in_kwargs = (v[1:] - v[:-1])
                break
        active_mask = self._enforce_keep_one_per_segment(active_mask, lengths_in_kwargs)

        # save original for restore
        setattr(module, "_ls_orig_h", hidden_states)
        setattr(module, "_ls_active_mask", active_mask)

        if active_mask.sum().item() < seq_len:
            kept_cnt = int(active_mask.sum().item())

            new_kwargs = {}
            for k, v in kwargs.items():
                new_kwargs[k] = self._recursive_prune(v, active_mask, seq_len, name_hint=f"kwargs.{k}")
            kwargs = new_kwargs

            new_args = []
            for i, arg in enumerate(args):
                new_args.append(self._recursive_prune(arg, active_mask, seq_len, name_hint=f"args[{i}]"))
            args = tuple(new_args)

            self._active_pairs.setdefault(layer_idx, []).append((kept_cnt, seq_len))

        return args, kwargs

    def forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        orig_h = getattr(module, "_ls_orig_h", None)
        active_mask = getattr(module, "_ls_active_mask", None)

        if hasattr(module, "_ls_orig_h"):
            delattr(module, "_ls_orig_h")
        if hasattr(module, "_ls_active_mask"):
            delattr(module, "_ls_active_mask")

        if orig_h is None or active_mask is None:
            return output
        if active_mask.all():
            return output

        layer_out = output[0] if isinstance(output, tuple) else output
        full_out = orig_h.clone().to(layer_out.device)

        if full_out.dim() == 2:
            full_out[active_mask] = layer_out
        else:
            full_out[:, active_mask, :] = layer_out

        if isinstance(output, tuple):
            return (full_out,) + output[1:]
        return full_out

    def attn_forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        layer_idx = getattr(module, "layer_idx", 0)
        start_layer, end_layer = self._flexible_range(module)

        # decide rule
        if self.decide_once_in_flexible:
            if layer_idx != start_layer:
                return output
        else:
            if not (start_layer <= layer_idx < end_layer):
                return output
            if self.decide_interval and ((layer_idx - start_layer) % self.decide_interval != 0):
                return output

        attn_out = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(attn_out):
            return output

        # get attn input
        attn_in = None
        if kwargs is not None:
            attn_in = kwargs.get("hidden_states", None)
        if attn_in is None and inputs is not None and len(inputs) > 0:
            attn_in = inputs[0]
        if attn_in is None or (not torch.is_tensor(attn_in)):
            return output

        # align shapes
        if attn_out.dim() == 2 and attn_in.dim() == 3:
            attn_out = attn_out.unsqueeze(0)
        if attn_in.dim() == 2 and attn_out.dim() == 3:
            attn_in = attn_in.unsqueeze(0)
        if attn_out.shape != attn_in.shape:
            return output

        score = self._dash_score_delta(attn_out, attn_in)  # (N,)
        seq_len = score.shape[0]

        drop_target = int(round(self.compression_ratio * seq_len))
        if drop_target <= 0:
            return output

        sorted_idx = torch.argsort(score)     # small delta => drop first
        drop_idx = sorted_idx[:drop_target]

        if self.stability_threshold > 0:
            drop_idx = drop_idx[score[drop_idx] <= self.stability_threshold]

        active_mask = torch.ones(seq_len, dtype=torch.bool, device=score.device)
        if drop_idx.numel() > 0:
            active_mask[drop_idx] = False

        # protect CLS/global
        for t in self.protected_token_indices:
            if 0 <= t < seq_len:
                active_mask[t] = True

        # enforce keep-one per packed segment if lengths/cuseqlens exists in this attn call
        lengths_here = None
        if kwargs is not None:
            for v in kwargs.values():
                if torch.is_tensor(v) and self._is_lengths_vec(v, seq_len):
                    lengths_here = v
                    break
                if torch.is_tensor(v) and self._is_cu_seqlens(v, seq_len):
                    lengths_here = (v[1:] - v[:-1])
                    break
        active_mask = self._enforce_keep_one_per_segment(active_mask, lengths_here)

        self._global_active_mask = active_mask
        print(f"[DeltaPressViT] Decided@L{layer_idx}: {seq_len} -> {int(active_mask.sum())}")

        return output


