# dlshogi USI engine — pre-compiled inference image
#
# Base: NVIDIA TensorRT 25.03 (TensorRT 10.8, CUDA 12.8, driver 570+)
# Supports RTX 4090 (Ada, sm_89) and RTX 5090 (Blackwell, sm_120)
#
# Two-step build — keeps the checkpoint OUT of the Docker build context
# so the context stays small (~50 MB) and fully cacheable.
#
# Step 1: build the engine image (no model yet)
#   docker build -t hmatsuya/dlshogi-usi:latest -f vastai/Dockerfile .
#
# Step 2: export checkpoint → ONNX locally, then inject into the image
#   bash vastai/inject_model.sh \
#     dlshogi/wandb/wcsc36/wkpn9zzz/checkpoints/last.ckpt \
#     hmatsuya/dlshogi-usi:latest
#
# Push:
#   docker push hmatsuya/dlshogi-usi:latest

FROM nvcr.io/nvidia/tensorrt:25.03-py3 AS builder

# ── System packages ────────────────────────────────────────────────────────────
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Acquire::AllowInsecureRepositories "true";' > /etc/apt/apt.conf.d/99insecure && \
    echo 'Acquire::AllowDowngradeToInsecureRepositories "true";' >> /etc/apt/apt.conf.d/99insecure && \
    apt-get clean && \
    apt-get update -qq --allow-insecure-repositories && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99insecure

# ── Python dependencies ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    onnx==1.17.0 \
    cshogi==0.8.9 \
    numpy==2.2.5 \
    cython==3.0.11 \
    setuptools==79.0.1

# ── Copy source (no checkpoint — kept out of build context) ───────────────────
WORKDIR /build
COPY . .

# ── Build the cppshogi Python extension ───────────────────────────────────────
RUN pip install --no-cache-dir -e . --no-build-isolation

# ── Compile USI engine ─────────────────────────────────────────────────────────
WORKDIR /build/usi
RUN make -j"$(nproc)"

# ── Download Mafu opening book (Apery format v11, v5.0 release) ───────────────
RUN curl -fsSL \
    "https://github.com/mafu-opening-theory/Shogi_opening_theory/archive/refs/tags/v5.0.tar.gz" \
    | tar -xzf - \
        --strip-components=2 \
        -C /tmp \
        "Shogi_opening_theory-5.0/まふ定跡（Apery形式）ver11/book.bin" \
    && mv /tmp/book.bin /tmp/mafu_book.bin


# ══════════════════════════════════════════════════════════════════════════════
# Final image — binary + book only; model.onnx injected separately
# ══════════════════════════════════════════════════════════════════════════════
FROM nvcr.io/nvidia/tensorrt:25.03-py3

# USI engine binary
COPY --from=builder /build/usi/bin/usi /usr/local/bin/usi

# Mafu opening book
COPY --from=builder /tmp/mafu_book.bin /workspace/model/book.bin

# Convenience launch wrapper
COPY vastai/run_usi.sh /workspace/run_usi.sh
RUN chmod +x /workspace/run_usi.sh

# Placeholder — model.onnx is injected by inject_model.sh after the build
RUN mkdir -p /workspace/model

WORKDIR /workspace

# No ENTRYPOINT — Vast.ai SSH mode
