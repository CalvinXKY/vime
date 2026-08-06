import base64
import io
import json
import logging
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    # Some multi-modal models (e.g. Qwen3-Omni) ship the chat template in a
    # separate ``chat_template.json`` file rather than inside
    # ``tokenizer_config.json``. AutoTokenizer does not load it, so the
    # resulting tokenizer has ``chat_template=None`` and
    # ``tokenizer.apply_chat_template`` raises. Fall back to the standalone
    # file (and then to the processor) so downstream code that calls
    # ``tokenizer.apply_chat_template`` keeps working.
    if getattr(tokenizer, "chat_template", None) is None:
        import os

        ct_path = os.path.join(name_or_path, "chat_template.json")
        if os.path.isfile(ct_path):
            try:
                with open(ct_path) as f:
                    ct_data = json.load(f)
                chat_template = ct_data.get("chat_template") if isinstance(ct_data, dict) else None
                if chat_template:
                    tokenizer.chat_template = chat_template
            except Exception as e:
                logger.warning(f"Failed to load chat_template.json from {name_or_path}: {e}")
    return tokenizer


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids as lists), "pt" for modality-specific outputs
    result["text_kwargs"] = {
        **result.get("text_kwargs", {}),
        "return_tensors": None,
        "return_mm_token_type_ids": False,
    }
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    # WhisperFeatureExtractor (used by Qwen3-Omni) expects raw_speech as
    # list[np.ndarray] of 1D float arrays at 16 kHz. process_vision_info
    # returns audio as list[(np.ndarray, sample_rate)] tuples so the
    # rollout encoder can write WAV with the correct sample rate. Strip
    # the sample_rate before handing to the HF processor (we assume 16 kHz
    # — Qwen3-Omni's feature_extractor default — and rely on the processor
    # to resample if needed).
    audio_value = result.get("audio")
    if isinstance(audio_value, list) and audio_value:
        stripped = []
        saw_tuple = False
        for item in audio_value:
            if isinstance(item, tuple) and len(item) == 2 and hasattr(item[0], "shape"):
                stripped.append(item[0])
                saw_tuple = True
            else:
                stripped.append(item)
        if saw_tuple:
            result["audio"] = stripped

    return result


def _try_load_glm4v_processor(name_or_path: str, **kwargs):
    """Fallback: manually construct a Glm4vProcessor for GLM-4.6V / GLM-4.5V models.

    AutoProcessor fails for these models on transformers < 5.0 because
    the Glm46VProcessor / Glm4vMoeProcessor classes are not registered.
    The underlying Glm4vProcessor (non-MoE) works for both variants since
    they share the same vision architecture.
    """
    try:
        from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
        from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
    except ImportError:
        return None

    pp_path = Path(name_or_path) / "preprocessor_config.json"
    vp_path = Path(name_or_path) / "video_preprocessor_config.json"
    if not pp_path.exists():
        return None

    skip_keys = {"image_processor_type", "processor_class", "video_processor_type"}
    with open(pp_path) as f:
        pp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
    image_processor = Glm4vImageProcessor(**pp_cfg)

    video_processor = None
    if vp_path.exists():
        with open(vp_path) as f:
            vp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
        video_processor = Glm4vVideoProcessor(**vp_cfg)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    proc = Glm4vProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    logger.info(f"Loaded Glm4vProcessor manually for {name_or_path}")
    return proc


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer instead of a proper processor, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        # Fallback: try to construct a GLM-4.6V / GLM-4.5V processor manually.
        proc = _try_load_glm4v_processor(name_or_path, **kwargs)

    return proc


def _extract_images_from_messages(messages):
    """Extract PIL images from chat messages containing multimodal content.

    Handles base64 strings (with or without data: URI prefix), file paths,
    and PIL Image objects embedded in message content dicts.
    """
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_data = item.get("image")
            if image_data is None:
                continue
            if isinstance(image_data, Image.Image):
                images.append(image_data)
            elif isinstance(image_data, str):
                if image_data.startswith("data:"):
                    _, encoded = image_data.split(",", 1)
                    images.append(Image.open(io.BytesIO(base64.b64decode(encoded))))
                else:
                    try:
                        raw = base64.b64decode(image_data)
                        images.append(Image.open(io.BytesIO(raw)))
                    except Exception:
                        # Not base64 — try as file path
                        images.append(Image.open(image_data))
    return images


def _extract_audios_from_messages(messages):
    """Extract audio samples from chat messages containing multimodal content.

    Returns a list of ``(np.ndarray, sample_rate)`` tuples. Supports:
    - dict with ``audio`` key (file path or URL string)
    - dict with ``audio_url`` key (URL/data URL string)
    - tuple ``(np.ndarray, sample_rate)`` already loaded
    - np.ndarray (assumes 16 kHz)
    """
    audios = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "audio":
                continue
            audio_data = item.get("audio") or item.get("audio_url")
            if audio_data is None:
                continue
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                # (np.ndarray, sample_rate) — pass through
                audios.append(audio_data)
            elif isinstance(audio_data, str):
                if audio_data.startswith("data:"):
                    # data URL — decode inline
                    try:
                        _, encoded = audio_data.split(",", 1)
                        raw = base64.b64decode(encoded)
                        audio, sr = _load_audio_bytes(raw)
                        if audio is not None:
                            audios.append((audio, sr))
                    except Exception as e:
                        logger.warning(f"Failed to decode data-URL audio: {e}")
                elif audio_data.startswith(("http://", "https://", "file://")):
                    # URL — return as-is so the rollout server can fetch it
                    audios.append(audio_data)
                else:
                    # File path — load locally
                    try:
                        audio, sr = _load_audio_file(audio_data)
                        if audio is not None:
                            audios.append((audio, sr))
                    except Exception as e:
                        logger.warning(f"Failed to load audio file {audio_data}: {e}")
            elif hasattr(audio_data, "shape"):
                # np.ndarray — assume 16 kHz (Qwen3-Omni default)

                audios.append((audio_data, 16000))
    return audios


def _resample_to_16k(audio, sr):
    """Resample audio to 16 kHz mono to match vLLM preprocessing."""
    TARGET_SR = 16000
    if sr != TARGET_SR:
        # resample_poly uses integer up/down ratios
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), TARGET_SR)
        up = TARGET_SR // g
        down = int(sr) // g
        audio = resample_poly(audio, up, down).astype("float32")
        sr = TARGET_SR
    return audio, sr


def _load_audio_bytes(raw: bytes):
    """Decode audio bytes (WAV/MP3/FLAC) into (np.ndarray, sample_rate) at 16 kHz."""
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # mono mixdown
        audio, sr = _resample_to_16k(audio, sr)
        return audio, sr
    except Exception as e:
        logger.warning(f"soundfile failed to decode audio: {e}")
        return None, None


def _load_audio_file(path: str):
    """Load audio from a local file path into (np.ndarray, sample_rate) at 16 kHz."""
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio, sr = _resample_to_16k(audio, sr)
        return audio, sr
    except Exception as e:
        logger.warning(f"soundfile failed to read {path}: {e}")
        return None, None


def process_vision_info(prompt, processor):
    """Extract PIL images, videos, and audios from the message list for training.

    Tries qwen_vl_utils first (Qwen VL family — does not support audio),
    then falls back to generic extraction for other models (e.g. GLM-4.6V,
    Qwen3-Omni). Audio is always extracted via the generic path since
    qwen_vl_utils 0.0.14 does not support audio.

    Returns a dict with keys ``images``, ``videos``, ``audio``. Note that
    ``audio`` is singular to match ``Qwen3OmniMoeProcessor.__call__``'s
    parameter name (transformers processor parameter names are not consistent
    across models — Qwen2Audio uses ``audios``, Qwen3-Omni uses ``audio``).
    """
    audios = _extract_audios_from_messages(prompt) or None

    try:
        from qwen_vl_utils import process_vision_info as qwen_process_vision_info

        if hasattr(processor.image_processor, "patch_size"):
            image_patch_size = processor.image_processor.patch_size
        else:
            image_patch_size = DEFAULT_PATCH_SIZE
        images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    except Exception:
        # Fallback: generic extraction for non-Qwen models
        images = _extract_images_from_messages(prompt) or None
        videos = None

    return {"images": images, "videos": videos, "audio": audios}


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def encode_audio_for_rollout_engine(audio) -> str:
    """Encode audio into a WAV data URL for the vLLM render API.

    Accepts either:
    - ``(np.ndarray, sample_rate)`` tuple — encode to WAV base64
    - ``str`` URL (http/https/file) — return as-is so the server fetches it
    """
    if isinstance(audio, str):
        return audio
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise ValueError(f"Unsupported audio type: {type(audio)}; expected tuple or URL str")
    audio_arr, sr = audio
    try:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, sr, format="WAV", subtype="FLOAT")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:audio/wav;base64,{audio_base64}"
    except Exception as e:
        raise RuntimeError(f"Failed to encode audio for rollout engine: {e}") from e


def encode_video_for_rollout_engine(video) -> str:
    """Encode video frames as an MJPEG data URL for the vLLM render API.

    vLLM's video media IO treats ``data:video/jpeg;base64,FRAME1,FRAME2,...`` as
    a list of JPEG frames and does not re-sample them, which keeps train/infer
    ``grid_thw`` aligned with the HF processor frames.

    Accepts:
    - ``str`` URL — returned as-is
    - ``torch.Tensor`` of shape ``(num_frames, C, H, W)`` (values in ``[0, 1]``)
    - ``list`` of PIL Images
    """
    if isinstance(video, str):
        return video

    import numpy as np
    import torch

    if isinstance(video, torch.Tensor):
        frames = video.detach().cpu().float()
        if frames.dim() == 4 and frames.shape[1] in (1, 3):
            frames = frames.permute(0, 2, 3, 1)  # (N, H, W, C)
        frames = (frames.clamp(0, 1).numpy() * 255).astype(np.uint8)
        encoded = []
        for frame in frames:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG")
            encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return f"data:video/jpeg;base64,{','.join(encoded)}"

    if isinstance(video, list) and video:
        encoded = []
        for item in video:
            if not isinstance(item, Image.Image):
                raise ValueError(f"Unsupported video frame type: {type(item)}")
            buf = io.BytesIO()
            item.save(buf, format="JPEG")
            encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return f"data:video/jpeg;base64,{','.join(encoded)}"

    raise ValueError(f"Unsupported video type: {type(video)}")
