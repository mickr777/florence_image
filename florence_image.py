import os
os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")

import sys
import threading
from contextlib import contextmanager
from typing import Literal

import torch
from PIL import Image as PILImage
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BartTokenizerFast,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    PreTrainedModel,
)
from transformers import __version__ as HF_VERSION
from transformers.configuration_utils import PreTrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.utils.hub import cached_file

from invokeai.invocation_api import (
    BaseInvocation,
    InvocationContext,
    invocation,
    InputField,
    StringOutput,
    ImageField,
)


_TRANSFORMERS_MAJOR = int(HF_VERSION.split(".", 1)[0])
_FLORENCE_LOAD_LOCK = threading.RLock()


def _uses_transformers_v5() -> bool:
    return _TRANSFORMERS_MAJOR >= 5


@contextmanager
def _legacy_florence_config_compat():
    """Keep Florence-2's legacy config contract working on Transformers 5.x."""
    if not _uses_transformers_v5():
        yield
        return

    original_post_init = PreTrainedConfig.__post_init__
    generation_defaults = GenerationConfig._get_default_generation_params()

    def compat_post_init(config, **kwargs):
        preserved = {name: kwargs[name] for name in generation_defaults if name in kwargs}
        original_post_init(config, **kwargs)

        # Transformers 5 moved these off PreTrainedConfig. Florence's remote
        # config still reads them directly, so preserve values from config.json.
        for name, default in generation_defaults.items():
            if name in preserved:
                setattr(config, name, preserved[name])
            elif not hasattr(config, name):
                setattr(config, name, default)

        # These were implicit PretrainedConfig defaults in Transformers 4.x and
        # are still read by Florence's legacy model code.
        if not hasattr(config, "tie_word_embeddings"):
            config.tie_word_embeddings = True
        if not hasattr(config, "torchscript"):
            config.torchscript = False

    PreTrainedConfig.__post_init__ = compat_post_init
    try:
        yield
    finally:
        PreTrainedConfig.__post_init__ = original_post_init


@contextmanager
def _legacy_florence_model_base_compat():
    """Restore the old weight-tying helper removed in Transformers 5.x."""
    if not _uses_transformers_v5() or hasattr(PreTrainedModel, "_tie_or_clone_weights"):
        yield
        return

    def _tie_or_clone_weights(model, output_embeddings, input_embeddings):
        if getattr(model.config, "torchscript", False):
            output_embeddings.weight = torch.nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        bias = getattr(output_embeddings, "bias", None)
        if bias is not None:
            bias.data = torch.nn.functional.pad(
                bias.data,
                (0, output_embeddings.weight.shape[0] - bias.shape[0]),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    PreTrainedModel._tie_or_clone_weights = _tie_or_clone_weights
    try:
        yield
    finally:
        delattr(PreTrainedModel, "_tie_or_clone_weights")


class _FlorenceTorchProxy:
    """Use CPU only for legacy torch.linspace calls that would otherwise land on meta."""

    def __init__(self, torch_module):
        self._torch = torch_module

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def linspace(self, *args, **kwargs):
        result = self._torch.linspace(*args, **kwargs)
        if getattr(getattr(result, "device", None), "type", None) == "meta" and kwargs.get("device") is None:
            safe_kwargs = dict(kwargs)
            safe_kwargs["device"] = "cpu"
            return self._torch.linspace(*args, **safe_kwargs)
        return result


@contextmanager
def _legacy_florence_torch_compat(model_cls):
    if not _uses_transformers_v5():
        yield
        return

    module = sys.modules.get(model_cls.__module__)
    original_torch = getattr(module, "torch", None) if module is not None else None
    if module is None or original_torch is None:
        yield
        return

    module.torch = _FlorenceTorchProxy(original_torch)
    try:
        yield
    finally:
        module.torch = original_torch


def _patch_remote_florence_model(model_cls) -> None:
    """Patch only the loaded Florence remote module for Transformers 5.x."""
    if not _uses_transformers_v5():
        return

    module = sys.modules.get(model_cls.__module__)
    if module is None:
        return

    language_base = getattr(module, "Florence2LanguagePreTrainedModel", None)
    if language_base is not None:
        original_init_weights = getattr(language_base, "_init_weights", None)
        if callable(original_init_weights) and not getattr(original_init_weights, "_invokeai_florence_guard", False):
            def guarded_init_weights(self, target_module):
                own_params = list(target_module.parameters(recurse=False))
                own_buffers = [buffer for buffer in target_module.buffers(recurse=False) if buffer is not None]
                if own_params or own_buffers:
                    params_loaded = all(getattr(param, "_is_hf_initialized", False) for param in own_params)
                    buffers_loaded = all(getattr(buffer, "_is_hf_initialized", False) for buffer in own_buffers)
                    if params_loaded and buffers_loaded:
                        return
                return original_init_weights(self, target_module)

            guarded_init_weights._invokeai_florence_guard = True
            language_base._init_weights = guarded_init_weights

    tied_weight_maps = {
        "Florence2LanguageModel": {
            "encoder.embed_tokens.weight": "shared.weight",
            "decoder.embed_tokens.weight": "shared.weight",
        },
        "Florence2LanguageForConditionalGeneration": {
            "model.encoder.embed_tokens.weight": "model.shared.weight",
            "model.decoder.embed_tokens.weight": "model.shared.weight",
            "lm_head.weight": "model.shared.weight",
        },
        "Florence2ForConditionalGeneration": {
            "language_model.model.encoder.embed_tokens.weight": "language_model.model.shared.weight",
            "language_model.model.decoder.embed_tokens.weight": "language_model.model.shared.weight",
            "language_model.lm_head.weight": "language_model.model.shared.weight",
        },
    }

    # Older Florence forks rebuild the top-level tied-weight metadata inside
    # __init__ as a list immediately before calling self.post_init(). That
    # overwrites the class-level mapping above. Transformers 5 expects a dict
    # and calls .keys()/.values() on it, so normalize it at the last possible
    # point before the Transformers post-init machinery runs.
    original_top_post_init = getattr(model_cls, "post_init", None)
    if callable(original_top_post_init) and not getattr(
        original_top_post_init, "_invokeai_florence_tied_guard", False
    ):
        def compat_top_post_init(self):
            if isinstance(getattr(self, "_tied_weights_keys", None), (list, tuple)):
                self._tied_weights_keys = tied_weight_maps["Florence2ForConditionalGeneration"].copy()
            return original_top_post_init(self)

        compat_top_post_init._invokeai_florence_tied_guard = True
        model_cls.post_init = compat_top_post_init
    for class_name, mapping in tied_weight_maps.items():
        cls = getattr(module, class_name, None)
        if cls is not None and isinstance(getattr(cls, "_tied_weights_keys", None), (list, tuple)):
            cls._tied_weights_keys = mapping

    # Transformers 5 treats these as writable class capabilities. Several older
    # Florence forks expose them as read-only @property methods instead, which
    # causes "property ... has no setter" during model initialization. Shadow the
    # legacy properties directly on both the base and concrete class. We force
    # eager attention below, so advertising SDPA/Flash support is unnecessary.
    top_base = getattr(module, "Florence2PreTrainedModel", None)
    for cls in (top_base, model_cls):
        if cls is not None:
            cls._supports_sdpa = False
            cls._supports_flash_attn_2 = False


def _load_florence_tokenizer_v5(model_name: str, cache_dir: str):
    """Load Florence's BART tokenizer without parsing legacy special-token metadata."""
    tokenizer_file = cached_file(model_name, "tokenizer.json", cache_dir=cache_dir)
    if tokenizer_file is None:
        raise RuntimeError(f"{model_name} does not provide tokenizer.json")

    # Some Florence fine-tunes (notably gokaygokay) ship a legacy
    # special_tokens_map.json whose additional_special_tokens are dictionaries.
    # Transformers 5 rejects that format. Loading tokenizer.json directly keeps
    # the trained vocabulary/merges/added-token IDs while avoiding the obsolete
    # metadata file; Florence2Processor registers the required special tokens.
    tokenizer = BartTokenizerFast(
        tokenizer_file=tokenizer_file,
        model_max_length=1024,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        mask_token="<mask>",
    )
    tokenizer.additional_special_tokens = []
    return tokenizer


def _load_florence_processor(model_name: str, cache_dir: str):
    if not _uses_transformers_v5():
        return AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    image_processor = CLIPImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = _load_florence_tokenizer_v5(model_name, cache_dir)

    processor_cls = get_class_from_dynamic_module(
        "processing_florence2.Florence2Processor",
        model_name,
        cache_dir=cache_dir,
    )
    return processor_cls(image_processor=image_processor, tokenizer=tokenizer)


def _load_florence_model(model_name: str, cache_dir: str, device: torch.device, use_cuda: bool):
    if not _uses_transformers_v5():
        if use_cuda:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                device_map={"": 0},
            )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(device)

    with _FLORENCE_LOAD_LOCK:
        model_cls = get_class_from_dynamic_module(
            "modeling_florence2.Florence2ForConditionalGeneration",
            model_name,
            cache_dir=cache_dir,
        )
        _patch_remote_florence_model(model_cls)

        load_kwargs = {
            "cache_dir": cache_dir,
            "attn_implementation": "eager",
        }
        if use_cuda:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = {"": 0}

        with (
            _legacy_florence_config_compat(),
            _legacy_florence_model_base_compat(),
            _legacy_florence_torch_compat(model_cls),
        ):
            model = model_cls.from_pretrained(model_name, **load_kwargs)

    if not use_cuda:
        model = model.to(device)
    return model


def _prepare_florence_image(processor, image):
    original_size = (image.width, image.height)
    processor_kwargs = {}

    if _uses_transformers_v5():
        image_processor = getattr(processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            height = size.get("height")
            width = size.get("width")
        else:
            height = getattr(size, "height", None)
            width = getattr(size, "width", None)

        if isinstance(width, int) and isinstance(height, int):
            if (image.width, image.height) != (width, height):
                image = image.resize((width, height), resample=PILImage.Resampling.BICUBIC)
            processor_kwargs["do_resize"] = False

    return image, processor_kwargs, original_size

def _ensure_generate(obj) -> None:
    if callable(getattr(obj, "generate", None)):
        return
    Patched = type(f"{obj.__class__.__name__}Gen", (obj.__class__, GenerationMixin), {})
    obj.__class__ = Patched

def _build_gen_cfg(model, processor) -> GenerationConfig:
    try:
        gen_cfg = GenerationConfig.from_model_config(model.config)
    except Exception:
        gen_cfg = GenerationConfig()
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        if getattr(gen_cfg, "eos_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            gen_cfg.eos_token_id = tok.eos_token_id
        if getattr(gen_cfg, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
            gen_cfg.pad_token_id = tok.pad_token_id
    if getattr(gen_cfg, "transformers_version", None) is None:
        gen_cfg.transformers_version = HF_VERSION
    return gen_cfg

def _patch_model_for_generation(model, processor) -> None:
    _ensure_generate(model)
    for name in ("language_model", "text_model", "model", "lm"):
        sub = getattr(model, name, None)
        if sub is not None:
            _ensure_generate(sub)

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        gen_cfg = _build_gen_cfg(model, processor)
    else:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            if getattr(gen_cfg, "eos_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                gen_cfg.eos_token_id = tok.eos_token_id
            if getattr(gen_cfg, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
                gen_cfg.pad_token_id = tok.pad_token_id
    model.generation_config = gen_cfg
    for name in ("language_model", "text_model", "model", "lm"):
        sub = getattr(model, name, None)
        if sub is not None:
            try:
                sub.generation_config = gen_cfg
            except Exception:
                pass

@invocation(
    "Image_Description_Florence2",
    title="Image Description Using Florence 2",
    tags=["image", "caption", "florence2"],
    category="vision",
    version="0.5.3",
    use_cache=False,
)
class FlorenceImageCaptionInvocation(BaseInvocation):
    """Generates a description for an input image using Florence 2."""

    input_image: ImageField = InputField(description="An image to describe")

    caption_type: Literal["Caption", "Detailed Caption", "More Detailed Caption"] = (
        InputField(description="Select the type of caption", default="Caption")
    )

    model_type: Literal[
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
        "gokaygokay/Florence-2-Flux-Large",
        "gokaygokay/Florence-2-SD3-Captioner",
        "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
        "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
        "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
        "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
    ] = InputField(
        description="Select the type of model", default="microsoft/Florence-2-base"
    )

    prepend_text: str = InputField(description="Text to prepend to the prompt", default="")
    append_text: str = InputField(description="Text to append to the prompt", default="")

    def describe_image(
        self, context: InvocationContext, image, caption_type, prepend_text, append_text
    ):
        try:
            context.util.signal_progress("Preparing to load the model...")
            model_name = self.model_type
            folder_name = model_name.replace("/", "-")
            cache_dir = os.path.join(os.path.dirname(__file__), "models", folder_name)
            os.makedirs(cache_dir, exist_ok=True)

            context.util.signal_progress(f"Loading {model_name} model from cache")
            processor = _load_florence_processor(model_name, cache_dir)

            use_cuda = torch.cuda.is_available()
            use_mps = torch.backends.mps.is_available()
            device = torch.device("cuda:0" if use_cuda else ("mps" if use_mps else "cpu"))
            model = _load_florence_model(model_name, cache_dir, device, use_cuda)

            _patch_model_for_generation(model, processor)

            if getattr(image, "mode", None) != "RGB":
                context.util.signal_progress("Converting image to RGB mode.")
                image = image.convert("RGB")

            task_prompt = {
                "Caption": "<CAPTION>",
                "Detailed Caption": "<DETAILED_CAPTION>",
                "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
            }[caption_type]

            image, processor_kwargs, original_image_size = _prepare_florence_image(processor, image)
            inputs = processor(
                text=task_prompt,
                images=image,
                return_tensors="pt",
                **processor_kwargs,
            )
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            if self.model_type.startswith("MiaoshouAI/Florence-2-large-PromptGen"):
                inputs.pop("attention_mask", None)

            if device.type == "cuda" and getattr(model, "dtype", None) == torch.float16 and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            context.util.signal_progress("Generating caption...")
            gen_cfg = getattr(model, "generation_config", None)
            eos_id = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
            pad_id = getattr(gen_cfg, "pad_token_id", None) if gen_cfg else None

            try:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    generation_config=gen_cfg,
                )
            except AttributeError:
                lm = getattr(model, "language_model", None)
                if lm is None or not callable(getattr(lm, "generate", None)):
                    raise
                generated_ids = lm.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    generation_config=getattr(lm, "generation_config", gen_cfg),
                )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            parsed = processor.post_process_generation(
                generated_text, task=task_prompt, image_size=original_image_size
            )
            caption = parsed.get(task_prompt, "") if isinstance(parsed, dict) else str(parsed)

            final_caption = f"{prepend_text} {caption} {append_text}".strip()
            context.util.signal_progress("Caption generation complete.")
            return final_caption

        except Exception as e:
            raise RuntimeError(f"Error during image description: {str(e)}") from e

        finally:
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded and memory cleared.")

    def invoke(self, context: InvocationContext) -> StringOutput:
        try:
            pil_image = context.images.get_pil(self.input_image.image_name)
            description = self.describe_image(
                context,
                pil_image,
                self.caption_type,
                self.prepend_text,
                self.append_text,
            )
            return StringOutput(value=description)
        except Exception as e:
            context.util.signal_progress(f"Error occurred: {str(e)}")
            return StringOutput(value=f"Error: {str(e)}")
