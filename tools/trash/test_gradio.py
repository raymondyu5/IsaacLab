import gradio as gr
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import os

# Load Processor & VLA Model
processor = AutoProcessor.from_pretrained("openvla/openvla-7b",
                                          trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).to("cuda:0")


# Prediction function
def predict_action(prompt, image_path):
    try:
        if not prompt or not image_path:
            return {"error": "Missing prompt or image_path"}

        # Check if the image file exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        # Load and process the image
        image = Image.open(image_path)
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

        # Predict action
        action = vla.predict_action(**inputs,
                                    unnorm_key="bridge_orig",
                                    do_sample=False)

        return {"action": action.tolist()}
    except Exception as e:
        return {"error": str(e)}


# Create Gradio interface with an API-compatible setup
app = gr.Interface(
    fn=predict_action,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt..."),
        gr.Textbox(label="Image Path",
                   placeholder="Enter the full image path...")
    ],
    outputs=gr.JSON(label="Output"),
)

# Launch the Gradio app
if __name__ == "__main__":
    app.launch(share=False, server_name="localhost", server_port=8555)
