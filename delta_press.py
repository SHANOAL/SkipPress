# delta_press.py
import logging
from dataclasses import dataclass,field
from typing import Optional, Dict, Tuple, List, Literal
import torch
from torch import nn
from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)




# delta_press_vit.py
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Literal, Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class DeltaPressViT:
    """
    Qwen2-VL Vision Token Pruning (Computation Skipping) - Scheme A (DASH-style).

    Key fix for Qwen2VL-7B vision packed attention:
    - Vision attention may pack multiple segments and uses `lengths` (or `cu_seqlens`)
      to split Q/K/V along seq dim.
    - If we prune seq tokens, we MUST update lengths/cuseqlens accordingly,
      otherwise torch.split will crash.

    Decision (Scheme A):
    - Use DASH score on attention sublayer output U (pre-residual):
        score_t = ||U_t||_2
      Drop tokens with smallest score first.
    """

    # NOTE: here compression_ratio means "drop fraction" (drop_ratio), not keep_ratio
    compression_ratio: float = 0.5

    # layer range (as ratios of total layers)
    start_ratio: float = 0.3
    end_ratio: float = 1.0

    # kept for compatibility (not used)
    metric: Literal["cos", "l1", "l2"] = "l2"

    # decide policy
    decide_once_in_flexible: bool = True
    decide_interval: int = 0  # if decide_once_in_flexible=False, decide every k layers when k>0

    # optional gating: only drop if score <= threshold when >0
    stability_threshold: float = 0.0

    # always keep these token indices
    protected_token_indices: Tuple[int, ...] = (0,)

    # runtime states
    _global_active_mask: Optional[torch.Tensor] = None
    _active_pairs: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    _vit_total_layers: Optional[int] = None

    def __post_init__(self):
        self._active_pairs = {}

    # ------------------------ helpers ------------------------
    def _flexible_range(self, module: nn.Module) -> Tuple[int, int]:
        total_layers = self._vit_total_layers
        if total_layers is None:
            cfg = getattr(module, "config", None)
            total_layers = (
                getattr(cfg, "num_hidden_layers", None)
                or getattr(cfg, "num_layers", None)
                or getattr(cfg, "depth", None)
                or 32
            )
        start_layer = int(total_layers * self.start_ratio)
        end_layer = int(total_layers * self.end_ratio)
        end_layer = max(start_layer + 1, end_layer)
        return start_layer, end_layer

    @staticmethod
    def _dash_score_u(attn_out: torch.Tensor) -> torch.Tensor:
        """
        Scheme A (DASH): score_t = ||U_t||_2, where U is attention sublayer pre-residual output.
        attn_out: [B,N,H] or [N,H] -> score: [N] (mean over batch if needed)
        """
        s = attn_out.norm(dim=-1)  # [B,N] or [N]
        if s.dim() == 2:
            return s.mean(dim=0)   # [N]
        return s                   # [N]

    @staticmethod
    def _is_lengths_vec(x: torch.Tensor, ref_len: int) -> bool:
        # lengths: 1D int, sum == ref_len, all > 0
        if x.dim() != 1 or x.numel() < 1:
            return False
        if x.dtype not in (torch.int32, torch.int64, torch.long):
            return False
        if int(x.sum().item()) != ref_len:
            return False
        if torch.any(x <= 0):
            return False
        return True

    @staticmethod
    def _is_cu_seqlens(x: torch.Tensor, ref_len: int) -> bool:
        # cu_seqlens: 1D int, starts with 0, ends with ref_len, nondecreasing
        if x.dim() != 1 or x.numel() < 2:
            return False
        if x.dtype not in (torch.int32, torch.int64, torch.long):
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

    def _enforce_keep_one_per_segment(
        self, mask: torch.Tensor, lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Ensure each packed segment keeps at least 1 token; otherwise torch.split may fail.
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
        new_lens: List[int] = []
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
        new_lengths = self._update_lengths_by_mask(lengths, mask)
        new_cu = torch.zeros((new_lengths.numel() + 1,), device=cu.device, dtype=cu.dtype)
        new_cu[1:] = torch.cumsum(new_lengths, dim=0)
        return new_cu

    def _recursive_prune(self, data: Any, mask: torch.Tensor, ref_len: int, name_hint: str = "") -> Any:
        """
        Brutal recursive pruning:
        - Any tensor whose seq dimension matches ref_len will be pruned.
        - Special handling for `lengths` and `cu_seqlens` so split() won't crash.
        """
        if data is None:
            return None

        if isinstance(data, tuple):
            return tuple(
                self._recursive_prune(item, mask, ref_len, f"{name_hint}[{i}]")
                for i, item in enumerate(data)
            )

        if isinstance(data, list):
            return [
                self._recursive_prune(item, mask, ref_len, f"{name_hint}[{i}]")
                for i, item in enumerate(data)
            ]

        if isinstance(data, dict):
            return {
                k: self._recursive_prune(v, mask, ref_len, f"{name_hint}.{k}")
                for k, v in data.items()
            }

        if torch.is_tensor(data):
            # ---- Special: packed segment lengths ----
            if self._is_lengths_vec(data, ref_len):
                return self._update_lengths_by_mask(data, mask)

            # ---- Special: cu_seqlens ----
            if self._is_cu_seqlens(data, ref_len):
                return self._update_cu_seqlens_by_mask(data, mask)

            # common layouts
            if data.shape[0] == ref_len:
                return data[mask]

            if data.dim() >= 2 and data.shape[1] == ref_len:
                return data[:, mask]

            # masks like [B, heads, S, S] or similar
            if data.dim() >= 3 and data.shape[-1] == ref_len:
                try:
                    if data.shape[-2] == ref_len:
                        return data[..., mask, :][..., :, mask]
                    else:
                        return data[..., mask]
                except Exception:
                    pass

        return data

    # ------------------------ hooks ------------------------
    def forward_pre_hook(self, module: nn.Module, args, kwargs):
        """
        Attach with with_kwargs=True.
        Prune any tensor arguments that carry the packed seq dim (including lengths/cu_seqlens).
        """
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]
        if hidden_states is None or (not torch.is_tensor(hidden_states)):
            return args, kwargs

        # Qwen2VL vision typically (B, N, D) or sometimes (N, D)
        seq_len = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]

        layer_idx = getattr(module, "layer_idx", 0)
        start_layer, end_layer = self._flexible_range(module)
        in_flexible = start_layer <= layer_idx < end_layer

        active_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)

        # reuse global decision mask inside flexible range
        if in_flexible and self._global_active_mask is not None and self._global_active_mask.shape[0] == seq_len:
            active_mask = self._global_active_mask.to(hidden_states.device)

        # protect token indices (e.g., CLS)
        for t in self.protected_token_indices:
            if 0 <= t < seq_len:
                active_mask[t] = True

        # IMPORTANT: enforce keep-one per segment if lengths/cuseqlens exists
        lengths_in_kwargs = None
        for v in kwargs.values():
            if torch.is_tensor(v) and self._is_lengths_vec(v, seq_len):
                lengths_in_kwargs = v
                break
            if torch.is_tensor(v) and self._is_cu_seqlens(v, seq_len):
                lengths_in_kwargs = (v[1:] - v[:-1])
                break
        active_mask = self._enforce_keep_one_per_segment(active_mask, lengths_in_kwargs)

        # save for restore in post hook
        setattr(module, "_ls_orig_h", hidden_states)
        setattr(module, "_ls_active_mask", active_mask)

        # prune
        if int(active_mask.sum().item()) < seq_len:
            kept_cnt = int(active_mask.sum().item())

            new_kwargs = {k: self._recursive_prune(v, active_mask, seq_len, name_hint=f"kwargs.{k}")
                          for k, v in kwargs.items()}
            kwargs = new_kwargs

            new_args = [self._recursive_prune(arg, active_mask, seq_len, name_hint=f"args[{i}]")
                        for i, arg in enumerate(args)]
            args = tuple(new_args)

            self._active_pairs.setdefault(layer_idx, []).append((kept_cnt, seq_len))

        return args, kwargs

    def forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        """
        Rebuild full seq length output: inactive tokens are identity (keep orig hidden).
        """
        orig_h = getattr(module, "_ls_orig_h", None)
        active_mask = getattr(module, "_ls_active_mask", None)

        # cleanup
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
        """
        Scheme A decision hook.
        Attach to the vision attention module with with_kwargs=True.

        NOTE: This assumes attn_out is U (pre-residual attention output).
        score_t = ||U_t||_2, drop the smallest scores first.
        """
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

        # accept [N,H] or [B,N,H]
        if attn_out.dim() not in (2, 3):
            return output

        score = self._dash_score_u(attn_out)  # [N]
        seq_len = int(score.shape[0])

        drop_target = int(round(self.compression_ratio * seq_len))
        if drop_target <= 0:
            return output

        # drop smallest
        sorted_idx = torch.argsort(score, descending=False)
        drop_idx = sorted_idx[:drop_target]

        # optional gating: only drop if score <= threshold
        if self.stability_threshold > 0:
            drop_idx = drop_idx[score[drop_idx] <= self.stability_threshold]

        active_mask = torch.ones(seq_len, dtype=torch.bool, device=score.device)
        if drop_idx.numel() > 0:
            active_mask[drop_idx] = False

        # protect token indices
        for t in self.protected_token_indices:
            if 0 <= t < seq_len:
                active_mask[t] = True

        # enforce keep-one per packed segment (lengths/cu_seqlens)
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
        print(f"[DeltaPressViT][SchemeA] Decided@L{layer_idx}: {seq_len} -> {int(active_mask.sum())}")

        return output



