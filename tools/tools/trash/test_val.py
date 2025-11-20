import threading
import socket
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import os
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Shutdown event to signal server stop
shutdown_event = threading.Event()

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
@app.post("/predict")
async def predict_action(prompt: str = Form(...), image: UploadFile = None):
    try:
        if not prompt or not image:
            raise HTTPException(status_code=400,
                                detail="Missing prompt or image")

        # Save the uploaded file temporarily
        temp_file_path = f"/tmp/{image.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await image.read())

        # Load and process the image
        img = Image.open(temp_file_path)
        inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)

        # Predict action
        action = vla.predict_action(**inputs,
                                    unnorm_key="bridge_orig",
                                    do_sample=False)

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Signal shutdown after the first request
        shutdown_event.set()

        return {"action": action.tolist()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Root endpoint for health check
@app.get("/")
def root():
    return {"message": "Server is running"}


# Middleware to handle shutdown
@app.on_event("shutdown")
def shutdown():
    print("Server shutting down...")


def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external server to get the IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":

    def run_server():
        local_ip = get_local_ip()
        try:
            # Fetch public IP
            import requests
            public_ip = requests.get("https://api.ipify.org").text
        except Exception:
            public_ip = "Unavailable"

        print(f"Server is running and accessible at:")
        print(f"  Local: http://127.0.0.1:8080")
        print(f"  Network: http://{local_ip}:8080")
        print(
            f"  Public: http://{public_ip}:8080 (if port forwarding is enabled)"
        )

        uvicorn.run(app, host="0.0.0.0", port=8080)

    # Start the server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
