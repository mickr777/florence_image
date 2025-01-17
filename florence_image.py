import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from invokeai.invocation_api import (
    BaseInvocation,
    InvocationContext,
    invocation,
    InputField,
    StringOutput,
    ImageField,
)
from typing import Literal


@invocation(
    "Image_Description_Florence2",
    title="Image Description Using Florence 2",
    tags=["image", "caption", "florence2"],
    category="vision",
    version="0.4.1",
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
    ] = InputField(
        description="Select the type of model", default="microsoft/Florence-2-base"
    )

    prepend_text: str = InputField(
        description="Text to prepend to the prompt", default=""
    )

    append_text: str = InputField(
        description="Text to append to the prompt", default=""
    )

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

            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if image.mode != "RGB":
                context.util.signal_progress("Converting image to RGB mode.")
                image = image.convert("RGB")

            task_prompt_map = {
                "Caption": "<CAPTION>",
                "Detailed Caption": "<DETAILED_CAPTION>",
                "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
            }
            task_prompt = task_prompt_map[caption_type]

            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(
                device
            )

            if model.dtype == torch.float16:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            context.util.signal_progress("Generating caption...")
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            parsed_answer = processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(image.width, image.height)
            )

            caption = (
                parsed_answer.get(task_prompt, "")
                if isinstance(parsed_answer, dict)
                else str(parsed_answer)
            )

            final_caption = f"{prepend_text} {caption} {append_text}".strip()
            context.util.signal_progress("Caption generation complete.")
            return final_caption

        except Exception as e:
            raise RuntimeError(f"Error during image description: {str(e)}") from e

        finally:
            if "model" in locals():
                del model
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
            print(f"Generated Description: {description}")
            return StringOutput(value=description)
        except Exception as e:
            context.util.signal_progress(f"Error occurred: {str(e)}")
            return StringOutput(value=f"Error: {str(e)}")
