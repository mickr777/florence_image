import os
os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")

import torch
from typing import Literal

from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation import GenerationMixin, GenerationConfig
from transformers import __version__ as HF_VERSION

from invokeai.invocation_api import (
    BaseInvocation,
    InvocationContext,
    invocation,
    InputField,
    StringOutput,
    ImageField,
)

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

    gen_cfg = _build_gen_cfg(model, processor)
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
    version="0.4.8",
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
            processor = AutoProcessor.from_pretrained(
                model_name, cache_dir=cache_dir, trust_remote_code=True
            )

            use_cuda = torch.cuda.is_available()
            use_mps = torch.backends.mps.is_available()

            if use_cuda:
                device = torch.device("cuda:0")
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    device_map={"": 0},
                )
            else:
                device = torch.device("mps" if use_mps else "cpu")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                ).to(device)

            if not hasattr(model, "_supports_sdpa"):
                model._supports_sdpa = False

            _patch_model_for_generation(model, processor)

            if getattr(image, "mode", None) != "RGB":
                context.util.signal_progress("Converting image to RGB mode.")
                image = image.convert("RGB")

            task_prompt = {
                "Caption": "<CAPTION>",
                "Detailed Caption": "<DETAILED_CAPTION>",
                "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
            }[caption_type]

            inputs = processor(text=task_prompt, images=image, return_tensors="pt")
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

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
                generated_text, task=task_prompt, image_size=(image.width, image.height)
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
