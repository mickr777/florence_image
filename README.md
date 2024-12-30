# Florence 2 Image Caption

A node for InvokeAI that uses the Florence 2 model family to generate image descriptions. This node can generate captions, detailed captions, or highly detailed captions based on user selection.

#### Fields:

| Fields         | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| input_image    | The input image to be described.                                            |
| caption_type   | Select the type of caption: "Caption", "Detailed Caption", or "More Detailed Caption". |
| model_type     | Select the Florence model variant to use. Options include:                  |
|                | - `microsoft/Florence-2-base`                                              |
|                | - `microsoft/Florence-2-large`                                             |
|                | - `gokaygokay/Florence-2-Flux-Large`                                       |
|                | - `gokaygokay/Florence-2-SD3-Captioner`                                    |
|                | - `MiaoshouAI/Florence-2-base-PromptGen-v1.5`                              |
|                | - `MiaoshouAI/Florence-2-large-PromptGen-v1.5`                             |

#### Info:
- On first use of a model, the required model files will be automatically downloaded and cached in the `models` directory on the node.

#### Notes:
- The generated captions vary in detail based on the selected `caption_type`.
- Supports custom models compatible with the Florence architecture available on Hugging Face.
