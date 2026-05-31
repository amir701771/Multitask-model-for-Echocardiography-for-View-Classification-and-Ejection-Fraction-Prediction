# Step-by-Step HuggingFace Spaces Deployment Guide

This project is preconfigured with the required metadata and server layouts for **one-click deployment** on **HuggingFace Spaces** (using the Streamlit SDK). Follow this step-by-step guide to launch your live clinical diagnostic web application.

---

## 🚀 Deployment Instructions

### Step 1: Create a Space on HuggingFace
1. Log in or sign up at [Hugging Face](https://huggingface.co/).
2. Click on **Spaces** in the top navigation bar, then click **Create new Space** (or go to [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Fill in the Space settings:
   - **Space Name**: e.g., `multitask-echocardiography-diagnostics`
   - **License**: Select `MIT` (matching this project's license).
   - **Select the Space SDK**: Click on **Streamlit**.
   - **Space Hardware**: Select the free **CPU basic** instance (the application is lightweight and does not require a GPU for demonstration runs).
   - **Visibility**: Select **Public** or **Private** based on your preference.
4. Click **Create Space**.

---

### Step 2: Connect and Push to HuggingFace
HuggingFace Spaces are backed by Git repositories. You can connect your local workspace and push the code directly.

1. Open your terminal in the root directory of this project.
2. Add your HuggingFace Space repository as a new Git remote named `hf`:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   ```
   *(Replace `<your-username>` and `<your-space-name>` with your actual HuggingFace details).*

3. Push the main branch to HuggingFace:
   ```bash
   git push -f hf main
   ```
   *Note: HuggingFace may prompt you for your username and your **HuggingFace Access Token** as the password. You can generate a token in your HuggingFace profile settings under "Access Tokens" (write permission required).*

---

### Step 3: Verify and View the Live Application
1. Go back to your Space page on the HuggingFace browser tab.
2. You will see the Space status transition to **Building** as it reads `requirements.txt` and compiles python dependencies.
3. Within 1-2 minutes, the status will change to **Running**, and the Streamlit clinical interface will appear live!

---

## 🛠️ Deployment Features

- **No Heavy Weight Constraints**: Medical model weights (`.pth.tar` files) are heavy and excluded from Git. The Streamlit app detects this and automatically runs in a high-fidelity **Clinical Simulation Mode** on the cloud. This allows recruiters to immediately test case inputs and view prediction graphs without hitting file upload/storage bounds.
- **Preconfigured Theme**: Custom `.streamlit/config.toml` configurations are automatically read by the space runner to theme the container with clinical styles matching our design system.
- **System-Agnostic File Handling**: Uploaded video assets are written into the container's system temp files, making the app immune to directory permission warnings.
