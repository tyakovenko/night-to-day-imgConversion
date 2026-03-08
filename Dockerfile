# HF Spaces — Docker SDK
# Uses a venv (non-root) to isolate pip installs from system Python.

FROM python:3.11-slim

# Non-root user matching HF Spaces uid convention
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN chown -R user /home/user
USER user

# Virtual environment — all pip installs go here, never into root site-packages
RUN python -m venv /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH" \
    # Disable Gradio analytics ping — prevents startup hang if outbound network is slow
    GRADIO_ANALYTICS_ENABLED=False \
    # Make Gradio port/host explicit so Docker health check can find the app
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Install CPU-only torch first (large; own layer so Docker can cache it)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
