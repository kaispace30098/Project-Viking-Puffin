# Project-Viking-Puffin
End-to-end project fine-tuning Stable Diffusion XL with DreamBooth and LoRA (PEFT) to generate images of a custom character.
# Project Viking Puffin: An AI-Powered Icelandic Adventure

![Generated Image of Kaiffin on an icy cliff in Iceland]
<img width="1024" height="1024" alt="manual_validation_test (3)" src="https://github.com/user-attachments/assets/0df3b634-c06e-4511-a502-8b38f8abd102" />

*An AI-generated image: "A photo of Kaiffin the Puffin on an icy cliff in Iceland"*

## The Backstory

In August 2025, my wife and I took an unforgettable trip to Iceland. On our very last day, we found a small, charming puffin souvenir wearing a Viking helmet. We fell in love with it instantly.

As we left, my wife said, "I wish we had bought this on the first day. We could have taken pictures with it everywhere we went."

That sparked an idea. As an AI Engineer, if we couldn't go back in time, could I use technology to create the memories our little puffin missed?

This project was born from that question. **The goal: to teach a state-of-the-art AI model (Stable Diffusion XL) to recognize our unique Viking Puffin—whom I named "Kaiffin"—so we could generate images of it in any scene imaginable, completing the Icelandic adventure it never had.**

## Tech Stack

* **Python**
* **PyTorch**
* **Hugging Face Diffusers & Accelerate**
* **Model:** Stable Diffusion XL 1.0
* **Fine-Tuning:** DreamBooth + LoRA (Low-Rank Adaptation)
* **Environment:** Google Colab (NVIDIA A100 GPU)

## End-to-End Workflow

1.  **Data Collection:** Took 20 high-resolution photos of the "Kaiffin" puffin from various angles with a smartphone.
2.  **Data Preprocessing:** Used an online tool (BIRME) to crop each photo into a square aspect ratio and resize it for training (512 x 512).
3.  **Cloud-Based Training:** Set up a Google Colab environment with an A100 GPU and necessary libraries. Fine-tuned the SDXL model using the prepared images and the DreamBooth/LoRA script.
4.  **Inference & Validation:** Wrote a separate script to load the trained LoRA weights and generate new images of "Kaiffin" based on text prompts.

## Technical Deep Dive: How It Works

This project leverages a technique called **Parameter-Efficient Fine-Tuning (PEFT)** to teach a massive, pre-trained model a new concept without altering the entire model. The key components are:

* **VAE (Variational Autoencoder):** This compresses high-resolution images into a smaller, information-rich "latent space." The diffusion process happens here, making it computationally efficient.
* **CLIP Text Encoders:** SDXL uses two of these to translate the text prompt (e.g., "a photo of Kaiffin") into a mathematical representation (embeddings) that guides the image generation process.
* **U-Net:** This is the core of the model. It iteratively removes noise from a random latent seed, gradually forming an image that matches the text prompt's guidance.
* **LoRA (Low-Rank Adaptation):** This is the magic. Instead of re-training the billions of parameters in the U-Net, we freeze them and inject tiny, trainable "adapter" layers. The resulting trained file is only a few megabytes, making it portable and efficient.

## How to Use

This repository contains two main files:

1.  **`training.ipynb`**: The notebook to fine-tune the SDXL model on your own set of images.
2.  **`inference.ipynb`**: The notebook to load your trained LoRA file and generate new images.

Follow the instructions within each notebook to run them.

## Future Work
* **Manually Implement the LoRA Mechanism:** Move beyond the high-level diffusers pipeline abstraction by building the core diffusion sampling loop (DDPM/DDIM) manually in PyTorch. and directly modify the U-Net's cross-attention layers to inject the LoRA weights. 
* **Deployment:** Wrap the inference pipeline in a simple web service (e.g., using Modal or FastAPI) to create an accessible endpoint for image generation.
