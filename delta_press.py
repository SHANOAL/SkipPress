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

# ============================================================
# DeltaPress: pruning is applied at decoder layer input.
# DASH-aligned (paper): decision uses delta-attn score Δ_t = ||U_t||_2
# Vision-only: only tokens in vision_mask are eligible to be dropped.
# ============================================================
@dataclass
class DeltaPress(BasePress):
    """
    Vision-only token pruning, DASH-aligned decision rule.

    Paper (DASH) logic:
      - At activation layer l_s: compute delta-attn score Δ_t = ||U_t||_2,
        where U is the attention sublayer pre-residual output.
      - Keep TopK tokens with largest Δ_t (equivalently: drop the smallest).
      - Halt dropped tokens in subsequent layers (prefill only).

    This implementation:
      - Applies the DASH score ONLY on eligible (vision) tokens.
      - Keeps all non-vision tokens always.
      - Makes decision once at start_layer (within flexible range) and reuses mask.
      - Does NOT rely on self_attn kwargs containing cache_position (robust for Qwen2-VL/HF).
    """

    # drop fraction among eligible vision tokens
    compression_ratio: float = 0.333
    start_ratio: float = 0.4
    end_ratio: float = 1.0

    # kept for compatibility (not used for DASH decision)
    metric: Literal["cos", "l1", "l2"] = "cos"
    decide_once_in_flexible: bool = True

    # optional; now interpreted as a score gate on ||U||_2 when > 0
    # (only drop if score <= threshold)
    stability_threshold: float = 0.0
    enable_timing: bool = False

    def __post_init__(self):
        assert 0.0 <= self.compression_ratio <= 1.0
        assert 0.0 <= self.start_ratio < self.end_ratio <= 1.0
        assert self.metric in ("cos", "l1", "l2")

        # global decision mask
        self._global_active_mask: Optional[torch.Tensor] = None
        self._decision_layer_idx: Optional[int] = None

        # vision mask (1D [S] bool on CPU)
        self._vision_mask_1d: Optional[torch.Tensor] = None

        # stats
        self._active_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self._timings_ms: Dict[int, float] = {}
        self._mem_deltas: Dict[int, int] = {}

    # ----------------------- public API -----------------------
    def set_vision_mask(self, vision_mask_1d: Optional[torch.Tensor]):
        """
        vision_mask_1d: [S] bool. True means visual token position(s).
        Only these positions are eligible to be dropped.
        """
        if vision_mask_1d is None:
            self._vision_mask_1d = None
            return
        if not torch.is_tensor(vision_mask_1d):
            raise TypeError("vision_mask_1d must be a torch.Tensor or None")
        if vision_mask_1d.dim() != 1:
            raise ValueError(f"vision_mask_1d must be 1D [S], got {vision_mask_1d.shape}")
        self._vision_mask_1d = vision_mask_1d.bool().detach().to("cpu")

    # ----------------------- helpers -----------------------
    def _vision_mask_on(self, seq_len: int, device) -> Optional[torch.Tensor]:
        if self._vision_mask_1d is None:
            return None
        if self._vision_mask_1d.numel() != seq_len:
            return None
        return self._vision_mask_1d.to(device)

    @staticmethod
    def _dash_score(attn_out: torch.Tensor) -> torch.Tensor:
        """
        DASH delta-attn score: Δ_t = ||U_t||_2
        attn_out: [B,S,H] -> score: [S] (mean over batch)
        """
        # [B,S]
        s = attn_out.norm(dim=-1)
        # [S]
        return s.mean(dim=0)

    def _is_prefill(self, cache_position, seq_len: int) -> bool:
        """True if still in prefill (best-effort)."""
        try:
            return int(cache_position.max().item()) < seq_len
        except Exception:
            return True

    def _flexible_range(self, module: nn.Module) -> Tuple[int, int]:
        cfg = getattr(module, "config", None)
        total_layers = getattr(cfg, "num_hidden_layers", 32)
        start_layer = int(total_layers * self.start_ratio)
        end_layer = int(total_layers * self.end_ratio)
        end_layer = max(start_layer + 1, end_layer)
        return start_layer, end_layer

    # ============================================================
    # A) Decoder layer hooks:
    #    1) Pass flags to attn hook (prefill) + (optional) layer input
    #    2) Apply pruning + rebuild full length
    # ============================================================
    def forward_pre_hook(self, module: nn.Module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", None)
        hidden_from_args = False
        if hidden_states is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
            hidden_from_args = True

        cache_position = kwargs.get("cache_position", None)
        if hidden_states is None or cache_position is None:
            return args, kwargs

        if cache_position.device != hidden_states.device:
            cache_position = cache_position.to(hidden_states.device)
            kwargs["cache_position"] = cache_position

        bsz, seq_len, _ = hidden_states.shape
        layer_idx = getattr(module, "layer_idx", None)

        start_layer, end_layer = self._flexible_range(module)
        in_flexible = (layer_idx is not None and start_layer <= layer_idx < end_layer)
        in_prefill = self._is_prefill(cache_position, seq_len)

        # Reset runtime states at start of a new prefill
        if layer_idx == 0 and in_prefill:
            self._global_active_mask = None
            self._decision_layer_idx = None

        # -----------------------------------------------------------
        # Pass prefill flag to self_attn so attn hook does NOT depend
        # on self_attn kwargs containing cache_position (robust for Qwen2-VL).
        # -----------------------------------------------------------
        if hasattr(module, "self_attn"):
            module.self_attn._press_in_prefill = bool(in_prefill)
            # optional: keep for debugging/compat
            module.self_attn._press_layer_input = hidden_states

        # decode stage: don't prune
        if not in_prefill:
            for n in ("_ls_orig_h", "_ls_seq_len", "_ls_active_mask"):
                if hasattr(module, n):
                    delattr(module, n)
            self._start_timing(module)
            return args, kwargs

        # save original input for reconstruction
        setattr(module, "_ls_orig_h", hidden_states)
        setattr(module, "_ls_seq_len", seq_len)

        # default keep all
        active_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)

        # reuse global mask once decided
        if self.decide_once_in_flexible and in_flexible and (self._global_active_mask is not None):
            if self._global_active_mask.shape[0] == seq_len:
                active_mask = self._global_active_mask.to(hidden_states.device)
            else:
                active_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)

        setattr(module, "_ls_active_mask", active_mask)

        # ---------- prune tokens ----------
        if active_mask.sum().item() < seq_len:
            new_hidden = hidden_states[:, active_mask, :]

            # replace args/kwargs
            if hidden_from_args:
                args = (new_hidden,) + tuple(args[1:])
                if "hidden_states" in kwargs:
                    kwargs.pop("hidden_states")
            else:
                kwargs["hidden_states"] = new_hidden

            # cache_position can be [S] or [B,S]
            if torch.is_tensor(cache_position):
                if cache_position.dim() == 1:
                    kwargs["cache_position"] = cache_position[active_mask]
                elif cache_position.dim() == 2:
                    kwargs["cache_position"] = cache_position[:, active_mask]

            # attention_mask crop
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is not None:
                if attn_mask.device != hidden_states.device:
                    attn_mask = attn_mask.to(hidden_states.device)
                if attn_mask.dim() == 4:        # [B, nh, S, S] or [B,1,S,S]
                    attn_mask = attn_mask[:, :, active_mask, :][:, :, :, active_mask]
                elif attn_mask.dim() == 3:      # [B, S, S]
                    attn_mask = attn_mask[:, active_mask, :][:, :, active_mask]
                elif attn_mask.dim() == 2:      # [B, S]
                    attn_mask = attn_mask[:, active_mask]
                kwargs["attention_mask"] = attn_mask

            # RoPE position_embeddings crop
            pos_emb = kwargs.get("position_embeddings", None)
            if pos_emb is not None:
                cos_pe, sin_pe = pos_emb
                dev = hidden_states.device
                cos_pe = cos_pe.to(dev)
                sin_pe = sin_pe.to(dev)

                if cos_pe.dim() == 2:           # [S, D]
                    cos_pe = cos_pe[active_mask]
                    sin_pe = sin_pe[active_mask]
                elif cos_pe.dim() == 3:         # [B, S, D]
                    cos_pe = cos_pe[:, active_mask]
                    sin_pe = sin_pe[:, active_mask]
                elif cos_pe.dim() == 4:         # [3, B, S, D]
                    cos_pe = cos_pe[:, :, active_mask, :]
                    sin_pe = sin_pe[:, :, active_mask, :]
                kwargs["position_embeddings"] = (cos_pe, sin_pe)

        kept = int(active_mask.sum().item())
        self._active_pairs.setdefault(layer_idx if layer_idx is not None else -1, []).append((kept, seq_len))
        self._start_timing(module)
        return args, kwargs

    def forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        # Clean up the reference/flag to avoid memory leaks
        if hasattr(module, "self_attn"):
            for n in ("_press_layer_input", "_press_in_prefill"):
                if hasattr(module.self_attn, n):
                    delattr(module.self_attn, n)

        orig_h = getattr(module, "_ls_orig_h", None)
        active_in = getattr(module, "_ls_active_mask", None)
        seq_len = getattr(module, "_ls_seq_len", None)
        layer_idx = getattr(module, "layer_idx", None)

        if orig_h is None or active_in is None or seq_len is None:
            return output
        if orig_h.dim() != 3 or orig_h.shape[1] != active_in.shape[0]:
            return output

        # unpack
        if isinstance(output, tuple):
            layer_out = output[0]
            rest = output[1:]
        else:
            layer_out = output
            rest = ()

        # rebuild full length: inactive positions keep identity
        full_out = orig_h.to(layer_out.device).clone()
        active_in = active_in.to(layer_out.device)
        full_out[:, active_in, :] = layer_out

        cache_position = kwargs.get("cache_position", None)
        in_prefill = self._is_prefill(cache_position, seq_len) if cache_position is not None else True

        # decode: reset
        if not in_prefill:
            self._global_active_mask = None
            self._decision_layer_idx = None

        self._end_timing(module, layer_idx if layer_idx is not None else -1)

        if isinstance(output, tuple):
            return (full_out,) + rest
        return full_out

    # ============================================================
    # B) Attention post-hook: decide mask based on DASH score ||U||_2
    # Vision-only: only drop among eligible vision tokens.
    # ============================================================
    def attn_forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        """
        Attach to layer.self_attn with with_kwargs=True.
        Decision is made at start_layer (once) during prefill.

        IMPORTANT: We do NOT rely on kwargs['cache_position'] here because
        many HF attention modules do not forward it.
        """
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return output

        in_prefill = getattr(module, "_press_in_prefill", True)
        if not in_prefill:
            return output

        # unpack attention output U (Delta)
        attn_out = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(attn_out):
            return output
        if attn_out.dim() == 2:       # [S,H]
            attn_out = attn_out.unsqueeze(0)
        if attn_out.dim() != 3:
            return output

        bsz, seq_len, _ = attn_out.shape

        start_layer, end_layer = self._flexible_range(module)
        in_flexible = (start_layer <= layer_idx < end_layer)

        # Only decide at start_layer if we haven't decided yet
        need_decide = (
            self.decide_once_in_flexible
            and in_flexible
            and (self._global_active_mask is None)
            and (layer_idx == start_layer)
        )
        if not need_decide:
            return output

        with torch.no_grad():
            # ------------------------------
            # DASH score: Δ_t = ||U_t||_2
            # ------------------------------
            score = self._dash_score(attn_out)  # [S], larger => more important => keep

            vision_mask = self._vision_mask_on(seq_len, attn_out.device)

            # default: keep all
            active_mask = torch.ones(seq_len, dtype=torch.bool, device=attn_out.device)

            if vision_mask is None:
                eligible = torch.arange(seq_len, device=attn_out.device)
            else:
                eligible = vision_mask.nonzero(as_tuple=False).flatten()

            eligible_len = int(eligible.numel())
            drop_target = int(round(self.compression_ratio * eligible_len)) if eligible_len > 0 else 0

            if drop_target > 0 and eligible_len > 0:
                score_elig = score[eligible]

                # drop smallest scores (least update => least important)
                sorted_idx = torch.argsort(score_elig, dim=0, descending=False)
                drop_idx = eligible[sorted_idx[:drop_target]]

                # optional gating: only drop if score <= threshold
                if self.stability_threshold > 0:
                    ok = score_elig[sorted_idx[:drop_target]] <= self.stability_threshold
                    drop_idx = drop_idx[ok]

                active_mask[drop_idx] = False

            # non-vision always kept
            if vision_mask is not None:
                active_mask[~vision_mask] = True

            self._global_active_mask = active_mask
            self._decision_layer_idx = layer_idx

            kept = int(active_mask.sum().item())
            print(
                f"[DeltaPress][DASH vision-only] decide@L={layer_idx} S={seq_len} "
                f"eligible={eligible_len} drop={drop_target} kept={kept}/{seq_len} ({kept/seq_len:.2%})"
            )

        return output

    # ----------------------- KV interface -----------------------
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values

    # ----------------------- reporting -----------------------
    @property
    def skip_ratio(self) -> float:
        if not self._active_pairs:
            return 0.0
        ratios = []
        for pairs in self._active_pairs.values():
            for a, s in pairs:
                if s > 0:
                    ratios.append(1.0 - a / s)
        return sum(ratios) / len(ratios) if ratios else 0.0

    def speedup_estimate(self) -> float:
        if not self._active_pairs:
            return 1.0
        avg_skip = self.skip_ratio
        flexible_ratio = max(0.0, self.end_ratio - self.start_ratio)
        effective_skip = avg_skip * flexible_ratio
        if effective_skip <= 0:
            return 1.0
        attn_speedup = 1.0 / ((1.0 - effective_skip) ** 2)
        ffn_speedup = 1.0 / (1.0 - effective_skip)
        return min(0.7 * attn_speedup + 0.3 * ffn_speedup, 10.0)

    def dump_layer_skip_summary(self):
        lines = ["[DeltaPress] === LayerSkip summary (prefill) ==="]
        for lidx in sorted(self._active_pairs.keys()):
            pairs = self._active_pairs[lidx]
            ratios = [a / s for (a, s) in pairs if s > 0]
            avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
            ms = self._timings_ms.get(lidx, 0.0)
            dmem = self._mem_deltas.get(lidx, 0)
            lines.append(
                f"  layer {lidx:02d}: avg_active_ratio={avg_ratio:.3f} "
                f"time~{ms:.1f} ms  dmem~{dmem/1024/1024:.2f} MB"
            )
        lines.append(
            f"  avg_skip_ratio={self.skip_ratio:.3%}  est_speedup~{self.speedup_estimate():.2f}x"
        )
        logger.info("\n".join(lines))

    # ----------------------- timing helpers -----------------------
    def _start_timing(self, module):
        if not self.enable_timing or not torch.cuda.is_available():
            return None
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        ev_start.record()
        setattr(module, "_ls_ev_start", ev_start)
        setattr(module, "_ls_ev_end", ev_end)
        setattr(module, "_ls_mem0", torch.cuda.memory_allocated())
        return ev_start

    def _end_timing(self, module, layer_idx: int):
        if not self.enable_timing or not torch.cuda.is_available():
            return
        ev_start = getattr(module, "_ls_ev_start", None)
        ev_end = getattr(module, "_ls_ev_end", None)
        if ev_start is None or ev_end is None:
            return
        ev_end.record()
        torch.cuda.synchronize()
        dt = ev_start.elapsed_time(ev_end)
        self._timings_ms[layer_idx] = self._timings_ms.get(layer_idx, 0.0) + float(dt)
        mem0 = getattr(module, "_ls_mem0", None)
        if mem0 is not None:
            mem1 = torch.cuda.memory_allocated()
            self._mem_deltas[layer_idx] = self._mem_deltas.get(layer_idx, 0) + int(mem1 - mem0)
        for n in ("_ls_ev_start", "_ls_ev_end", "_ls_mem0"):
            if hasattr(module, n):
                delattr(module, n)


# ============================================================
# DeltaPressLNR: Kept for compatibility/reference (unchanged)
# ============================================================
@dataclass
class DeltaPressLNR(BasePress):
    compression_ratio: float = 0.333
    start_ratio: float = 0.4
    end_ratio: float = 1.0

    metric: Literal["cos", "l1", "l2"] = "cos"
    decide_once_in_flexible: bool = True
    order: Literal["first", "second"] = "second"

    stability_threshold: float = 0.0
    enable_timing: bool = False

    def __post_init__(self):
        assert 0.0 <= self.compression_ratio <= 1.0
        assert 0.0 <= self.start_ratio < self.end_ratio <= 1.0
        assert self.metric in ("cos", "l1", "l2")
        assert self.order in ("first", "second")

        self._prev_block_delta: Optional[torch.Tensor] = None  # [B,S,H] CPU
        self._h_in_block: Optional[torch.Tensor] = None        # [B,S,H]
        self._attn_full_delta: Optional[torch.Tensor] = None   # [B,S,H]

        self._global_active_mask: Optional[torch.Tensor] = None
        self._decision_layer_idx: Optional[int] = None

        self._active_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self._timings_ms: Dict[int, float] = {}
        self._mem_deltas: Dict[int, int] = {}

        self._vision_mask_1d: Optional[torch.Tensor] = None

    def set_vision_mask(self, vision_mask_1d: Optional[torch.Tensor]):
        if vision_mask_1d is None:
            self._vision_mask_1d = None
            return
        if not torch.is_tensor(vision_mask_1d) or vision_mask_1d.dim() != 1:
            raise ValueError("vision_mask_1d must be a 1D torch.Tensor [S].")
        self._vision_mask_1d = vision_mask_1d.bool().detach().to("cpu")

    def _vision_mask_on(self, seq_len: int, device) -> Optional[torch.Tensor]:
        if self._vision_mask_1d is None:
            return None
        if self._vision_mask_1d.numel() != seq_len:
            return None
        return self._vision_mask_1d.to(device)

    def _is_prefill(self, cache_position, seq_len: int) -> bool:
        if cache_position is None or not torch.is_tensor(cache_position):
            return True
        try:
            return int(cache_position.min().item()) == 0
        except Exception:
            return True

    def _flexible_range(self, module: nn.Module) -> Tuple[int, int]:
        cfg = getattr(module, "config", None)
        total_layers = getattr(cfg, "num_hidden_layers", 32)
        start_layer = int(total_layers * self.start_ratio)
        end_layer = int(total_layers * self.end_ratio)
        end_layer = max(start_layer + 1, end_layer)
        return start_layer, end_layer

    @staticmethod
    def _cos_1m(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_n = torch.clamp(x.norm(dim=-1), min=eps)
        y_n = torch.clamp(y.norm(dim=-1), min=eps)
        cos = (x * y).sum(dim=-1) / (x_n * y_n)
        cos = torch.clamp(cos, -1.0, 1.0)
        s = 1.0 - cos
        return s.mean(dim=0) if s.dim() == 2 else s

    # ---------------- attention hooks ----------------
    def attn_forward_pre_hook(self, module: nn.Module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]

        cache_position = kwargs.get("cache_position", None)
        layer_idx = getattr(module, "layer_idx", None)
        if hidden_states is None or cache_position is None or layer_idx is None:
            return args, kwargs

        self._h_in_block = hidden_states

        bsz, seq_len, _ = hidden_states.shape
        start_layer, end_layer = self._flexible_range(module)
        in_flexible = start_layer <= layer_idx < end_layer
        in_prefill = self._is_prefill(cache_position, seq_len)

        if layer_idx == 0 and in_prefill:
            self._global_active_mask = None
            self._decision_layer_idx = None
            self._prev_block_delta = None

        if (not in_prefill) or (not in_flexible):
            return args, kwargs

        # apply global mask (if already decided)
        if self._global_active_mask is not None and self._global_active_mask.shape[0] == seq_len:
            active_mask = self._global_active_mask.to(hidden_states.device)
            kept = int(active_mask.sum().item())
            if kept < seq_len:
                new_hidden = hidden_states[:, active_mask, :]

                # update args/kwargs
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    args = (new_hidden,) + tuple(args[1:])
                    kwargs.pop("hidden_states", None)
                else:
                    kwargs["hidden_states"] = new_hidden

                # cache_position
                if cache_position.dim() == 1:
                    kwargs["cache_position"] = cache_position[active_mask]
                elif cache_position.dim() == 2:
                    kwargs["cache_position"] = cache_position[:, active_mask]

                # attention_mask
                attn_mask = kwargs.get("attention_mask", None)
                if attn_mask is not None:
                    if attn_mask.dim() == 4:
                        attn_mask = attn_mask[:, :, active_mask, :][:, :, :, active_mask]
                    elif attn_mask.dim() == 3:
                        attn_mask = attn_mask[:, active_mask, :][:, :, active_mask]
                    elif attn_mask.dim() == 2:
                        attn_mask = attn_mask[:, active_mask]
                    kwargs["attention_mask"] = attn_mask

                # position_embeddings
                pos_emb = kwargs.get("position_embeddings", None)
                if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
                    cos_pe, sin_pe = pos_emb
                    if torch.is_tensor(cos_pe) and torch.is_tensor(sin_pe):
                        if cos_pe.dim() == 2:
                            cos_pe = cos_pe[active_mask]
                            sin_pe = sin_pe[active_mask]
                        elif cos_pe.dim() == 3:
                            cos_pe = cos_pe[:, active_mask]
                            sin_pe = sin_pe[:, active_mask]
                        elif cos_pe.dim() == 4:
                            cos_pe = cos_pe[:, :, active_mask, :]
                            sin_pe = sin_pe[:, :, active_mask, :]
                        kwargs["position_embeddings"] = (cos_pe, sin_pe)

        return args, kwargs

    def attn_forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        hidden_in = self._h_in_block
        if hidden_in is None:
            return output

        layer_out = output[0] if isinstance(output, tuple) else output
        extra = output[1:] if isinstance(output, tuple) else ()

        bsz, seq_len, _ = hidden_in.shape

        if self._global_active_mask is not None and self._global_active_mask.shape[0] == seq_len:
            active_mask = self._global_active_mask.to(hidden_in.device)
        else:
            active_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_in.device)

        kept = int(active_mask.sum().item())

        # expand attn delta to full length
        full_attn_delta = torch.zeros_like(hidden_in)
        if torch.is_tensor(layer_out):
            if layer_out.dim() == 2:
                layer_out_b = layer_out.unsqueeze(0)
                full_attn_delta[:, active_mask, :] = layer_out_b
            elif layer_out.dim() == 3:
                if layer_out.shape[1] == seq_len:
                    full_attn_delta = layer_out
                else:
                    full_attn_delta[:, active_mask, :] = layer_out
        self._attn_full_delta = full_attn_delta

        return (full_attn_delta,) + extra if isinstance(output, tuple) else full_attn_delta

    # ---------------- FFN hooks ----------------
    def ffn_forward_pre_hook(self, module: nn.Module, args, kwargs):
        return args, kwargs

    def ffn_forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        h_in_block = self._h_in_block
        attn_full_delta = self._attn_full_delta
        if h_in_block is None or attn_full_delta is None:
            return output

        ffn_out = output[0] if isinstance(output, tuple) else output
        extra = output[1:] if isinstance(output, tuple) else ()

        bsz, seq_len, _ = h_in_block.shape
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return output

        if self._global_active_mask is not None and self._global_active_mask.shape[0] == seq_len:
            active_mask = self._global_active_mask.to(h_in_block.device)
        else:
            active_mask = torch.ones(seq_len, dtype=torch.bool, device=h_in_block.device)

        kept = int(active_mask.sum().item())
        self._active_pairs.setdefault(layer_idx, []).append((kept, seq_len))

        # expand ffn delta to full length
        full_ffn_delta = torch.zeros_like(h_in_block)
        if torch.is_tensor(ffn_out):
            if ffn_out.dim() == 2:
                full_ffn_delta[:, active_mask, :] = ffn_out.unsqueeze(0)
            elif ffn_out.dim() == 3:
                if ffn_out.shape[1] == seq_len:
                    full_ffn_delta = ffn_out
                else:
                    full_ffn_delta[:, active_mask, :] = ffn_out

        block_delta = attn_full_delta + full_ffn_delta
        block_out = h_in_block + block_delta

        start_layer, end_layer = self._flexible_range(module)
        in_flexible = start_layer <= layer_idx < end_layer

        cache_position = kwargs.get("cache_position", None)
        in_prefill = self._is_prefill(cache_position, seq_len)

        need_decide = (
            self.decide_once_in_flexible
            and in_flexible
            and in_prefill
            and (self._global_active_mask is None)
            and (layer_idx == start_layer)
        )

        if need_decide:
            with torch.no_grad():
                if self.order == "first":
                    stability = self._cos_1m(h_in_block, block_out)
                else:
                    if self._prev_block_delta is not None and self._prev_block_delta.shape == block_delta.shape:
                        stability = self._cos_1m(block_delta, self._prev_block_delta.to(block_delta.device))
                    else:
                        stability = self._cos_1m(h_in_block, block_out)

                vision_mask = self._vision_mask_on(seq_len, h_in_block.device)
                active = torch.ones(seq_len, dtype=torch.bool, device=h_in_block.device)

                if vision_mask is None:
                    eligible = torch.arange(seq_len, device=h_in_block.device)
                else:
                    eligible = vision_mask.nonzero(as_tuple=False).flatten()

                eligible_len = int(eligible.numel())
                drop_target = int(round(self.compression_ratio * eligible_len)) if eligible_len > 0 else 0

                if drop_target > 0 and eligible_len > 0:
                    stab_elig = stability[eligible]
                    sorted_idx = torch.argsort(stab_elig, dim=0, descending=False)
                    drop_idx = eligible[sorted_idx[:drop_target]]
                    active[drop_idx] = False

                if vision_mask is not None:
                    active[~vision_mask] = True

                self._global_active_mask = active
                self._decision_layer_idx = layer_idx
                kept2 = int(active.sum().item())
                print(
                    f"[DeltaPressLNR] decide(vision-only)@L={layer_idx} kept={kept2}/{seq_len} "
                    f"({kept2/seq_len:.2%}), eligible={eligible_len}, drop_target={drop_target}"
                )

        if self.order == "second":
            self._prev_block_delta = block_delta.detach().cpu()

        if not in_prefill:
            self._global_active_mask = None
            self._decision_layer_idx = None
            self._prev_block_delta = None

        return (full_ffn_delta,) + extra if isinstance(output, tuple) else full_ffn_delta

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values

    # timing helpers (optional)
    def _start_timing(self, module):
        if not self.enable_timing or not torch.cuda.is_available():
            return None
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        ev_start.record()
        setattr(module, "_dp_ev_start", ev_start)
        setattr(module, "_dp_ev_end", ev_end)
        setattr(module, "_dp_mem0", torch.cuda.memory_allocated())
        return ev_start

    def _end_timing(self, module, layer_idx: int):
        if not self.enable_timing or not torch.cuda.is_available():
            return
        ev_start = getattr(module, "_dp_ev_start", None)
        ev_end = getattr(module, "_dp_ev_end", None)
        if ev_start is None or ev_end is None:
            return
        ev_end.record()
        torch.cuda.synchronize()
        dt = ev_start.elapsed_time(ev_end)
        self._timings_ms[layer_idx] = self._timings_ms.get(layer_idx, 0.0) + float(dt)

        mem0 = getattr(module, "_dp_mem0", None)
        if mem0 is not None:
            mem1 = torch.cuda.memory_allocated()
            self._mem_deltas[layer_idx] = self._mem_deltas.get(layer_idx, 0) + int(mem1 - mem0)
        for n in ("_dp_ev_start", "_dp_ev_end", "_dp_mem0"):
            if hasattr(module, n):
                delattr(module, n)
@dataclass
class DeltaPressRandom(BasePress):
    """
    Random Vision-only token pruning (Baseline).
    直接继承 BasePress，完全独立于 DeltaPress。
    """
    compression_ratio: float = 0.5
    start_ratio: float = 0.4
    end_ratio: float = 1.0
    
    # 这里的 metric 只是一个标签，不再受 DeltaPress 的 assert 限制
    metric: str = "random" 
    decide_once_in_flexible: bool = True
    enable_timing: bool = False

    def __post_init__(self):
        # 只检查比例范围，不检查 metric
        assert 0.0 <= self.compression_ratio <= 1.0
        assert 0.0 <= self.start_ratio < self.end_ratio <= 1.0

        # 初始化状态
        self._global_active_mask: Optional[torch.Tensor] = None
        self._decision_layer_idx: Optional[int] = None
        self._vision_mask_1d: Optional[torch.Tensor] = None

        # 统计数据容器
        self._active_pairs: Dict[int, List[Tuple[int, int]]] = {}

    # ================= 接口 =================
    def set_vision_mask(self, vision_mask_1d: Optional[torch.Tensor]):
        if vision_mask_1d is None:
            self._vision_mask_1d = None
            return
        self._vision_mask_1d = vision_mask_1d.bool().detach().to("cpu")

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """KVPress 要求的接口，这里不做 KV Cache 压缩，原样返回"""
        return keys, values

    # ================= 辅助函数 =================
    def _vision_mask_on(self, seq_len: int, device) -> Optional[torch.Tensor]:
        if self._vision_mask_1d is None or self._vision_mask_1d.numel() != seq_len:
            return None
        return self._vision_mask_1d.to(device)

    def _is_prefill(self, cache_position, seq_len: int) -> bool:
        try:
            return int(cache_position.max().item()) < seq_len
        except Exception:
            return True

    def _flexible_range(self, module: nn.Module) -> Tuple[int, int]:
        cfg = getattr(module, "config", None)
        total_layers = getattr(cfg, "num_hidden_layers", 32)
        start_layer = int(total_layers * self.start_ratio)
        end_layer = int(total_layers * self.end_ratio)
        end_layer = max(start_layer + 1, end_layer)
        return start_layer, end_layer

    # ============================================================
    # [Pre-Hook] 执行剪枝：如果 mask 已定，裁切 input/mask/rope
    # ============================================================
    def forward_pre_hook(self, module: nn.Module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", None)
        hidden_from_args = False
        if hidden_states is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
            hidden_from_args = True

        cache_position = kwargs.get("cache_position", None)
        if hidden_states is None or cache_position is None:
            return args, kwargs

        if cache_position.device != hidden_states.device:
            cache_position = cache_position.to(hidden_states.device)
            kwargs["cache_position"] = cache_position

        bsz, seq_len, _ = hidden_states.shape
        layer_idx = getattr(module, "layer_idx", None)

        start_layer, end_layer = self._flexible_range(module)
        in_flexible = (layer_idx is not None and start_layer <= layer_idx < end_layer)
        in_prefill = self._is_prefill(cache_position, seq_len)

        # 每一轮 Prefill 开始时重置状态
        if layer_idx == 0 and in_prefill:
            self._global_active_mask = None
            self._decision_layer_idx = None

        # 给 Attn 模块打标记
        if hasattr(module, "self_attn"):
            module.self_attn._press_in_prefill = bool(in_prefill)

        # 非 Prefill 阶段不做任何操作
        if not in_prefill:
            for n in ("_ls_orig_h", "_ls_seq_len", "_ls_active_mask"):
                if hasattr(module, n): delattr(module, n)
            return args, kwargs

        # === 核心剪枝逻辑 ===
        # 1. 保存原始输入，用于 Post-Hook 恢复
        setattr(module, "_ls_orig_h", hidden_states)
        setattr(module, "_ls_seq_len", seq_len)

        # 2. 获取 Mask (如果还没生成，默认为全 1)
        active_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
        
        # 如果 Mask 已在 Start Layer 决定好了，且当前层在范围内，就应用它
        if self.decide_once_in_flexible and in_flexible and (self._global_active_mask is not None):
            if self._global_active_mask.shape[0] == seq_len:
                active_mask = self._global_active_mask.to(hidden_states.device)

        setattr(module, "_ls_active_mask", active_mask)

        # 3. 如果需要剪枝，修改输入参数
        if active_mask.sum().item() < seq_len:
            new_hidden = hidden_states[:, active_mask, :]

            if hidden_from_args:
                args = (new_hidden,) + tuple(args[1:])
                if "hidden_states" in kwargs: kwargs.pop("hidden_states")
            else:
                kwargs["hidden_states"] = new_hidden

            # 裁切 cache_position
            if torch.is_tensor(cache_position):
                if cache_position.dim() == 1:
                    kwargs["cache_position"] = cache_position[active_mask]
                elif cache_position.dim() == 2:
                    kwargs["cache_position"] = cache_position[:, active_mask]

            # 裁切 attention_mask (支持 2D/3D/4D)
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is not None:
                if attn_mask.device != hidden_states.device:
                    attn_mask = attn_mask.to(hidden_states.device)
                if attn_mask.dim() == 4:
                    attn_mask = attn_mask[:, :, active_mask, :][:, :, :, active_mask]
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask[:, active_mask, :][:, :, active_mask]
                elif attn_mask.dim() == 2:
                    attn_mask = attn_mask[:, active_mask]
                kwargs["attention_mask"] = attn_mask

            # 裁切 RoPE (Qwen2-VL 特殊格式)
            pos_emb = kwargs.get("position_embeddings", None)
            if pos_emb is not None:
                cos_pe, sin_pe = pos_emb
                dev = hidden_states.device
                cos_pe, sin_pe = cos_pe.to(dev), sin_pe.to(dev)

                if cos_pe.dim() == 2:
                    cos_pe, sin_pe = cos_pe[active_mask], sin_pe[active_mask]
                elif cos_pe.dim() == 3:
                    cos_pe, sin_pe = cos_pe[:, active_mask], sin_pe[:, active_mask]
                elif cos_pe.dim() == 4:
                    cos_pe, sin_pe = cos_pe[:, :, active_mask, :], sin_pe[:, :, active_mask, :]
                kwargs["position_embeddings"] = (cos_pe, sin_pe)

        kept = int(active_mask.sum().item())
        self._active_pairs.setdefault(layer_idx if layer_idx is not None else -1, []).append((kept, seq_len))
        return args, kwargs

    def forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        # 清理标记
        if hasattr(module, "self_attn") and hasattr(module.self_attn, "_press_in_prefill"):
            delattr(module.self_attn, "_press_in_prefill")

        orig_h = getattr(module, "_ls_orig_h", None)
        active_in = getattr(module, "_ls_active_mask", None)
        
        if orig_h is None or active_in is None:
            return output

        if isinstance(output, tuple):
            layer_out, rest = output[0], output[1:]
        else:
            layer_out, rest = output, ()

        # 恢复全长
        full_out = orig_h.to(layer_out.device).clone()
        active_in = active_in.to(layer_out.device)
        full_out[:, active_in, :] = layer_out

        if isinstance(output, tuple):
            return (full_out,) + rest
        return full_out

    def attn_forward_post_hook(self, module: nn.Module, inputs, kwargs, output):
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None: return output

        in_prefill = getattr(module, "_press_in_prefill", True)
        if not in_prefill: return output

        start_layer, end_layer = self._flexible_range(module)
        
        need_decide = (
            self.decide_once_in_flexible
            and (start_layer <= layer_idx < end_layer)
            and (self._global_active_mask is None)
            and (layer_idx == start_layer)
        )
        
        if not need_decide: return output
        
        attn_out = output[0] if isinstance(output, tuple) else output
        bsz, seq_len, _ = attn_out.shape
        device = attn_out.device

        with torch.no_grad():
            vision_mask = self._vision_mask_on(seq_len, device)
            active_mask = torch.ones(seq_len, dtype=torch.bool, device=device)

            if vision_mask is None:
                eligible = torch.arange(seq_len, device=device)
            else:
                eligible = vision_mask.nonzero(as_tuple=False).flatten()

            eligible_len = int(eligible.numel())
            drop_target = int(round(self.compression_ratio * eligible_len))

            if drop_target > 0 and eligible_len > 0:
                perm = torch.randperm(eligible_len, device=device)
                drop_indices = perm[:drop_target]
                drop_idx_in_seq = eligible[drop_indices]
                active_mask[drop_idx_in_seq] = False

            if vision_mask is not None:
                active_mask[~vision_mask] = True

            self._global_active_mask = active_mask
            self._decision_layer_idx = layer_idx

            kept = int(active_mask.sum().item())
            print(f"[DeltaPressRandom] Decided@L{layer_idx}: {kept}/{seq_len} tokens kept.")

        return output
