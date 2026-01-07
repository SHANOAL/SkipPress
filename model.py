from __future__ import annotations

import os
import sys
import warnings
import math
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import StoppingCriteria

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY
from .delta_press import DeltaPress,DeltaPressLNR ,DeltaPressViT, DeltaPressRandom# <<< 新增

VLLM_MAX_IMAGE_INPUT_NUM = 24


# =========================
# NEW: Latency / Prefill / FLOPs / KV Cache measurement utils
# =========================

@dataclass
class PerfStats:
    total_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    decode_tokens: int = 0
    ms_per_token: float = 0.0

    kv_cache_mb: float = 0.0
    peak_alloc_mb: float = 0.0
    peak_reserved_mb: float = 0.0

    prefill_flops: float = 0.0
    decode_flops: float = 0.0


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _bytes_per_elem(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == torch.float64:
        return 8
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return 4


def _kv_cache_size_mb(past_key_values) -> float:
    if past_key_values is None:
        return 0.0
    total_bytes = 0
    try:
        for layer_past in past_key_values:
            if layer_past is None:
                continue
            if isinstance(layer_past, (tuple, list)):
                for t in layer_past:
                    if torch.is_tensor(t):
                        total_bytes += t.numel() * _bytes_per_elem(t.dtype)
    except Exception:
        return 0.0
    return total_bytes / (1024.0 * 1024.0)


def _approx_transformer_flops(cfg, seq_len: int, new_tokens: int) -> tuple[float, float]:
    """
    Very rough FLOPs estimate (for trend comparison).
    Returns (prefill_flops, decode_flops).

    Assumptions (decoder-only):
      per-layer prefill ~ (4H^2 + 4HI)*S + 2S^2H
      per-layer decode for 1 token ~ (4H^2 + 4HI) + 2*S*H
    """
    if cfg is None:
        return 0.0, 0.0
    H = getattr(cfg, "hidden_size", None) or getattr(cfg, "dim", None)
    L = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    I = getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_hidden_size", None) or getattr(cfg, "n_inner", None)
    if H is None or L is None or I is None:
        return 0.0, 0.0

    S = int(seq_len)

    proj_prefill = 4.0 * H * H * S
    ffn_prefill = 4.0 * H * I * S
    attn_prefill = 2.0 * (S * S) * H
    prefill = L * (proj_prefill + ffn_prefill + attn_prefill)

    decode = 0.0
    for t in range(int(new_tokens)):
        St = S + t
        proj = 4.0 * H * H
        ffn = 4.0 * H * I
        attn = 2.0 * St * H
        decode += L * (proj + ffn + attn)

    return float(prefill), float(decode)


def _print_perf(prefix: str, stats: PerfStats):
    print(
        f"{prefix}[Perf] total={stats.total_ms:.2f}ms | "
        f"prefill={stats.prefill_ms:.2f}ms | "
        f"decode≈{stats.decode_ms:.2f}ms ({stats.decode_tokens} tok, {stats.ms_per_token:.2f} ms/tok) | "
        f"KV={stats.kv_cache_mb:.2f}MB | "
        f"peak_alloc={stats.peak_alloc_mb:.1f}MB peak_reserved={stats.peak_reserved_mb:.1f}MB | "
        f"FLOPs≈prefill={stats.prefill_flops/1e9:.2f}G decode={stats.decode_flops/1e9:.2f}G"
    )


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501

UNTIL = ["คาด"]


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        elif listinstr(['2.5', '2_5', 'qwen25', 'mimo'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag between `use_vllm` and `use_lmdeploy` to True"  # noqa: E501

        if self.use_vllm:
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=5,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )

        elif self.use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(session_len=32768, cache_max_entry_count=0.1, tp=num_gpus),
                chat_template_config=ChatTemplateConfig(model_name='qwen2d5-vl'))
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map="auto"
                # attn_implementation='flash_attention_2'  # if you have apex installed
            )
            self.model.eval()
            # self._init_layer_skip()

        torch.cuda.empty_cache()

    def _press_reset(self):
        press = getattr(self, "_delta_press", None)
        if press is None:
            return
        if hasattr(press, "_active_pairs"):
            press._active_pairs.clear()
        if hasattr(press, "_timings_ms"):
            press._timings_ms.clear()
        if hasattr(press, "_mem_deltas"):
            press._mem_deltas.clear()
    
    def _press_report(self, prefix=""):
        press = getattr(self, "_delta_press", None)
        if press is None:
            return
        try:
            if hasattr(press, "dump_layer_skip_summary"):
                press.dump_layer_skip_summary()
        except Exception as e:
            print(f"[PressReport] dump_layer_skip_summary failed: {e}")

        try:
            if hasattr(press, "skip_ratio") and hasattr(press, "speedup_estimate"):
                print(f"{prefix}[PressReport] skip_ratio={press.skip_ratio:.3%}, est_speedup={press.speedup_estimate():.2f}x")
        except Exception as e:
            print(f"[PressReport] skip_ratio/speedup failed: {e}")

    def _inject_vision_mask(self, inputs):
        press = getattr(self, "_delta_press", None)
        if press is None or (not hasattr(press, "set_vision_mask")):
            return

        tok = self.processor.tokenizer
        ids = inputs["input_ids"][0]  # [S] batch=1

        image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
        vision_mask = (ids == image_pad_id)

        try:
            video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
            vision_mask |= (ids == video_pad_id)
        except Exception:
            pass

        press.set_vision_mask(vision_mask)
        print(f"[vision_mask] seq={ids.numel()} vision={int(vision_mask.sum())}")

    def _init_layer_skip(self):
        # 只在 transformers 路径下使用；vLLM / lmdeploy 暂时不管
        if getattr(self, "use_vllm", False) or getattr(self, "use_lmdeploy", False):
            print("[LayerSkip] skip init: using vLLM / lmdeploy backend")
            return
    
        hf_model = self.model
        vlm_model = hf_model.model
        text_model = vlm_model.language_model
        text_config = text_model.config
        layers = text_model.layers
    
    #     # ============================================================
    #     # 方案 1（默认启用）：DeltaPressLNR（block 变化量判定；裁剪=attn/ffn 都不参与）
    #     # 挂法：attn pre/post + ffn pre/post
    #     # ============================================================
    #     press = DeltaPressLNR(
    #         compression_ratio=0.75,
    #         start_ratio=0,
    #         end_ratio=1.0,
    #         metric="l2",
    #         order="first",   # "first" / "second"
    #     )
    #     self._delta_press = press
    
    #     for i, layer in enumerate(layers):
    #         layer.layer_idx = i
    #         layer.config = text_config
    
    #         attn = layer.self_attn
    #         attn.layer_idx = i
    #         attn.config = text_config
    #         attn.register_forward_pre_hook(press.attn_forward_pre_hook, with_kwargs=True)
    #         attn.register_forward_hook(press.attn_forward_post_hook, with_kwargs=True)
    
    #         ffn = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None) or getattr(layer, "ffn", None)
    #         if ffn is not None:
    #             ffn.layer_idx = i
    #             ffn.config = text_config
    #             ffn.register_forward_pre_hook(press.ffn_forward_pre_hook, with_kwargs=True)
    #             ffn.register_forward_hook(press.ffn_forward_post_hook, with_kwargs=True)
    
    #     print(f"[LayerSkip] DeltaPressLNR initialized, num_layers={len(layers)}")
    #
        # ============================================================
        # 方案 2（想用 DeltaPress 就启用这个，把上面 LNR 那段注释掉）
        # 挂法：layer pre/post（裁剪+重建） + self_attn post（判定）
        # ============================================================
        press = DeltaPress(
            compression_ratio=0.65,
            start_ratio=0,
            end_ratio=1.0,
            metric="l2",  
        )
        self._delta_press = press
        
        for i, layer in enumerate(layers):
            layer.layer_idx = i
            layer.config = text_config
        
            # --------------------------------------------------------
            # 修改点在这里：直接使用 forward_pre_hook
            # 它内部已经包含了 "module.self_attn._press_layer_input = hidden_states" 的逻辑
            # --------------------------------------------------------
            layer.register_forward_pre_hook(press.forward_pre_hook, with_kwargs=True)
            
            # Layer Post Hook (恢复序列)
            layer.register_forward_hook(press.forward_post_hook, with_kwargs=True)
        
            attn = layer.self_attn
            attn.layer_idx = i
            attn.config = text_config
            
            # Attn Post Hook (决策计算)
            attn.register_forward_hook(press.attn_forward_post_hook, with_kwargs=True)
        
        print("[LayerSkip] DeltaPress (Input vs Input+Attn) initialized")

    # def _init_layer_skip(self):
    #     # 只在 transformers 路径下使用
    #     if getattr(self, "use_vllm", False) or getattr(self, "use_lmdeploy", False):
    #         print("[LayerSkip] skip init: using vLLM / lmdeploy backend")
    #         return

    #     hf_model = self.model
    #     vlm_model = hf_model.model
    #     text_model = vlm_model.language_model
    #     text_config = text_model.config
    #     layers = text_model.layers

    #     # ============================================================
    #     # 方案: Random Vision Pruning (Baseline)
    #     # ============================================================
    #     press = DeltaPressRandom(
    #         compression_ratio=0.8,  # 设置你的丢弃比例，例如 0.5 表示丢弃 50% 视觉 token
    #         start_ratio=0,        # 从第 40% 层开始丢弃 (例如 32层模型，从第12层开始)
    #         end_ratio=1.0,
    #         metric="random",        # 标记
    #     )
    #     self._delta_press = press

    #     for i, layer in enumerate(layers):
    #         layer.layer_idx = i
    #         layer.config = text_config

    #         # 1. Layer Pre Hook: 执行裁剪 (pruning)
    #         # 注意：DeltaPressRandom 继承自 DeltaPress，使用它的 forward_pre_hook
    #         layer.register_forward_pre_hook(press.forward_pre_hook, with_kwargs=True)
            
    #         # 2. Layer Post Hook: 执行恢复 (restoration)
    #         layer.register_forward_hook(press.forward_post_hook, with_kwargs=True)

    #         attn = layer.self_attn
    #         attn.layer_idx = i
    #         attn.config = text_config
            
    #         # 3. Attn Post Hook: 在 start_layer 做随机决策
    #         attn.register_forward_hook(press.attn_forward_post_hook, with_kwargs=True)

    #     print(f"[LayerSkip] DeltaPressRandom initialized (Ratio={press.compression_ratio}, Start={press.start_ratio})")

    # def _init_layer_skip(self):
    #     hf_model = self.model
    #     visual_model = getattr(hf_model.model, "visual", hf_model.model)
    #
    #     if hasattr(visual_model, "blocks"):
    #         blocks = visual_model.blocks
    #     elif hasattr(visual_model, "layers"):
    #         blocks = visual_model.layers
    #     else:
    #         print("[DeltaPressViT] Error: could not find ViT blocks (no .blocks / .layers)")
    #         return
    #
    #     vision_config = getattr(visual_model, "config", None)
    #
    #     press = DeltaPressViT(
    #         compression_ratio=0.8125,
    #         start_ratio=0.2,
    #         end_ratio=1.0,
    #         # compression_ratio=0.5,
    #         # start_ratio=0,
    #         # end_ratio=1.0,
    #         metric="l2",
    #         decide_once_in_flexible=True,
    #         stability_threshold=0.0,
    #         decide_interval=0,
    #         protected_token_indices=(0,)  # keep CLS/global token
    #     )
    #     press._vit_total_layers = len(blocks)
    #     self._delta_press = press
    #
    #     for i, blk in enumerate(blocks):
    #         blk.layer_idx = i
    #         blk.config = vision_config
    #
    #         blk.register_forward_pre_hook(press.forward_pre_hook, with_kwargs=True)
    #         blk.register_forward_hook(press.forward_post_hook, with_kwargs=True)
    #
    #         # qwen2vl visual block uses `attn` (你报错栈里就是 self.attn)
    #         attn_module = getattr(blk, "attn", None) \
    #             or getattr(blk, "self_attn", None) \
    #             or getattr(blk, "attention", None)
    #
    #         if attn_module is not None:
    #             attn_module.layer_idx = i
    #             attn_module.config = vision_config
    #             attn_module.register_forward_hook(press.attn_forward_post_hook, with_kwargs=True)
    #
    #     print(f"[DeltaPressViT] Initialized on qwen2vl vision blocks: layers={len(blocks)}")


    # def _inject_vision_mask(self, inputs):
    #     """
    #     重要修复：
    #     你原来从 input_ids 里找 <|image_pad|> 来保护 token，但那是“语言序列 token”，
    #     和 ViT patch token 序列长度一般不一致，几乎不会生效，还容易误导调试。
    #
    #     ViT token 保护已由 DeltaPressViT.protected_token_indices=(0,) 完成（保护 CLS）。
    #     这里留空/只做调试打印即可。
    #     """
    #     press = getattr(self, "_delta_press", None)
    #     if press is None:
    #         return
    #
    #     try:
    #         ids = inputs["input_ids"][0]
    #         print(f"[vision_mask] (noop) text_seq={ids.numel()} (ViT patch tokens protected by CLS keep)")
    #     except Exception:
    #         pass



    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value'])
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value'])
                    }
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                    if self.total_pixels is not None:
                        item['total_pixels'] = self.total_pixels
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
                    content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    # =========================
    # NEW: safe perf measurement compatible with kvpress/SDPA
    # =========================
    @torch.no_grad()
    def _measure_and_generate_safe(self, inputs, dataset=None) -> str:
        """
        Use a forward() for prefill measurement, then HF generate() for stable decoding.
        Avoids attention_mask / cache_position mismatch under kvpress patches.
        """
        stats = PerfStats()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            _cuda_sync()

        # ---- Prefill (forward only) ----
        _cuda_sync()
        t0 = _now_ms()
        out = self.model(**inputs, use_cache=True, return_dict=True)
        _cuda_sync()
        t1 = _now_ms()
        stats.prefill_ms = t1 - t0

        past = getattr(out, "past_key_values", None)
        stats.kv_cache_mb = _kv_cache_size_mb(past)

        # FLOPs (approx, trend)
        try:
            cfg = getattr(self.model, "config", None)
            seq_len = int(inputs["input_ids"].shape[1])
            max_new = int(self.generate_kwargs.get("max_new_tokens", self.max_new_tokens))
            pre, dec = _approx_transformer_flops(cfg, seq_len=seq_len, new_tokens=max_new)
            stats.prefill_flops = pre
            stats.decode_flops = dec
        except Exception:
            pass

        # ---- Total generate ----
        _cuda_sync()
        tg0 = _now_ms()
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        _cuda_sync()
        tg1 = _now_ms()
        stats.total_ms = tg1 - tg0

        prompt_len = int(inputs.input_ids.shape[1])
        gen_only = [o[prompt_len:] for o in generated_ids]
        stats.decode_tokens = int(gen_only[0].shape[0]) if len(gen_only) > 0 else 0

        # decode time (approx): stable for comparisons
        stats.decode_ms = max(0.0, stats.total_ms - stats.prefill_ms)
        stats.ms_per_token = (stats.decode_ms / stats.decode_tokens) if stats.decode_tokens > 0 else 0.0

        if torch.cuda.is_available():
            stats.peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            stats.peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)

        _print_perf(prefix=f"[{dataset}] " if dataset else "", stats=stats)

        out_text = self.processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0] if len(gen_only) > 0 else ""
        return out_text
    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info([messages], use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(text=text, images=images, audio=audios, videos=videos, padding=True, return_tensors='pt', use_audio_in_video=self.use_audio_in_video)  # noqa: E501
        else:
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')
        self._press_reset()
        self._inject_vision_mask(inputs)

        # ===== Safe perf measurement + generation (compatible with kvpress/SDPA) =====
        response = self._measure_and_generate_safe(inputs, dataset=dataset)

        self._press_report(prefix=f"[{dataset}] ")

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_lmdeploy(self, message, dataset=None):
        from lmdeploy import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            top_p=self.generate_kwargs['top_p'],
            top_k=self.generate_kwargs['top_k'],
            temperature=self.generate_kwargs['temperature'],
            repetition_penalty=self.generate_kwargs['repetition_penalty'],
        )
        gen_config.random_seed = None
        messages_list = self.message_to_lmdeploy(message, system_prompt=self.system_prompt)
        assert len(messages_list) == 1

        t0 = _now_ms()
        response = self.model(messages_list, gen_config=gen_config)[0]
        _cuda_sync()
        t1 = _now_ms()
        print(f"[{dataset}] [Perf-lmdeploy] total={t1 - t0:.2f}ms (lmdeploy backend: no prefill/kv/flops split here)")
        response = response.text
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            images, videos = process_vision_info(messages)
        print('finishing process vision info in vllm.')

        videos_nd = None
        video_inputs = None
        if DATASET_MODALITY(dataset) == 'VIDEO' and dataset is not None and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]

            video_inputs = {
                "prompt": text[0],
                "multi_modal_data": {"video": videos_nd[0]},
                "mm_processor_kwargs":{}
            }
            if self.use_audio_in_video:
                import vllm
                assert not vllm.envs.VLLM_USE_V1, ("V1 does not support use_audio_in_video. Please launch this example with `VLLM_USE_V1=0`.")  # noqa: E501
                video_inputs["multi_modal_data"]["audio"] = audios[0]
                video_inputs['mm_processor_kwargs']['use_audio_in_video'] = True
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print('video input sequence may be too long for vllm, Maybe cannot generate response for VLLM')

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.max_new_tokens, stop_token_ids=None
        )

        t0 = _now_ms()
        if images:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": images},
                },
                sampling_params=sampling_params,
            )
        elif videos_nd is not None:
            outputs = self.llm.generate(
                video_inputs,
                sampling_params=sampling_params,
            )
        else:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                },
                sampling_params=sampling_params,
            )
        _cuda_sync()
        t1 = _now_ms()
        print(f"[{dataset}] [Perf-vllm] total={t1 - t0:.2f}ms (vllm backend: no prefill/kv/flops split here)")

        generated_text = ""
        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        elif self.use_lmdeploy:
            return self.generate_inner_lmdeploy(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
class Qwen2VLChatAguvis(Qwen2VLChat):
    def __init__(self, mode=None, **kwargs):
        self.mode = mode
        super().__init__(**kwargs)
        self.processor.max_pixels = self.max_pixels
        self.processor.min_pixels = self.min_pixels

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        user_message = []
        for item in message:
            if "role" in item.keys():
                if item["role"] == "system":
                    self.system_prompt = item["value"]
                else:
                    item.pop("role")
                    user_message.append(item)
            else:
                user_message.append(item)
        message = user_message

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=CHAT_TEMPLATE,
        )
        # TODO: provide current action's low-level instruction
        # if False:
        #     # If low-level instruction is provided
        #     # We enforce using "Action: {low_level_instruction} to guide generation"
        #     recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
        if self.mode == "force-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
        elif self.mode == "force-plan-l1":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nAction: "
        elif self.mode == "force-plan-l3":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nObservation: "
        elif self.mode == "grounding":
            recipient_text = "<|im_start|>assistant<|recipient|>os\n"
        elif self.mode == "force-plan-free":
            recipient_text = "<|im_start|>assistant<|recipient|>all\n"
        elif self.mode == "self-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        text += recipient_text
        # print(text)

        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        self._press_reset()
        self._inject_vision_mask(inputs)

        # stop_str = "คาด"
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(
        #     keywords, self.processor.tokenizer, inputs.input_ids
        # )

        # ===== Safe perf measurement + generation (compatible with kvpress/SDPA) =====
        response = self._measure_and_generate_safe(inputs, dataset=dataset)

        self._press_report(prefix=f"[{dataset}] ")

        # for term in UNTIL:
        #     if len(term) > 0:
        #         response = response.split(term)[0]

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response

