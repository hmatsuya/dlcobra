# Vast.ai Championship Template — dlshogi USI Engine

Inference-only setup. The USI engine is **compiled at Docker image build time**, so instances start immediately with no compilation step.

## Quick Launch (for AI assistants)

Everything is already built and pushed. To launch an instance immediately:

```bash
# 1. Find cheapest RTX 4090/5090 with CUDA >= 12.8, excluding China (CN)
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cuda_max_good>=12.8 disk_space>=20 inet_down>=200' -o dph --raw \
  | python3 -c "
import sys, json
for o in json.load(sys.stdin):
    if ', CN' not in o.get('geolocation',''):
        print(f\"ID: {o['id']}  \${o['dph_total']:.3f}/hr  {o['geolocation']}\")
        break
"

# 2. Launch (replace OFFER_ID with ID from above)
vastai create instance <OFFER_ID> --image hmatsuya/dlshogi-usi:latest --disk 20 --ssh --direct

# 3. Wait for running, then get SSH details
vastai show instance <INSTANCE_ID> --raw | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['actual_status'])"
vastai ssh-url <INSTANCE_ID>
# → ssh://root@<HOST>:<PORT>

# 4. Test the engine (first run builds TRT cache, takes ~2 min)
ssh -o StrictHostKeyChecking=no -p <PORT> root@<HOST> \
  "echo -e 'usi\nisready\nusinewgame\nposition startpos\ngo byoyomi 3000\n' | timeout 180 /workspace/run_usi.sh 2>&1 | grep -E 'usiok|readyok|bestmove'"
# Expected: usiok → readyok → bestmove <move>

# 5. Stop billing when done
vastai destroy instance <INSTANCE_ID>
```

**Important**: always use `vastai destroy` (not `vastai stop`) when done — stopped instances still bill for storage. Always `destroy` + `create` (never `stop` + `start`) when deploying a new image.

**Current model**: `lzd2iw9l/last.ckpt` — exp032 KD fine-tuning run, step 71250, 65.7M params, sigmoid applied to value output.

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the engine image (source only — no model weights in build context) |
| `export_onnx.py` | Converts a pruned state dict or Lightning checkpoint to ONNX |
| `inject_model.sh` | Exports ONNX locally and injects it into the built image |
| `run_usi.sh` | Launch wrapper baked into the image at `/workspace/run_usi.sh` |
| `create_template.sh` | Creates (or updates) the Vast.ai template via CLI |
| `usi_ssh_proxy.ps1` | Windows PowerShell proxy — connects ShogiHome to the remote engine via SSH |
| `usi_ssh_proxy_ps.bat` | Thin `.bat` wrapper to launch the PowerShell proxy from ShogiHome |
| `usi_ssh_proxy.bat` | Self-contained batch alternative (no PowerShell required) |

## Build overview

The build is split into two steps to keep the Docker build context small (~50 MB).
The 250+ MB model weights are never sent through `docker build` — they are injected
separately after the image is built.

```
Step 1: docker build          Step 2: inject_model.sh
  source code only (~50 MB)     export ONNX locally → docker cp into image
  ↓                             ↓
  engine binary compiled        model.onnx committed into image layer
  book.bin downloaded           ↓
  ↓                           docker push
  image without model
```

## Step 1 — Build the engine image

Run from the **repo root**. No checkpoint needed — the build context is source code only.

```bash
docker build -t hmatsuya/dlshogi-usi:latest -f vastai/Dockerfile .
```

The build:
1. Installs Python deps and builds the `cppshogi` Cython extension
2. Compiles the USI engine with `make`
3. Downloads the Mafu opening book (Apery format v11)
4. Copies only the binary + book into the lean final image (multi-stage build)

All layers are fully cacheable. Rebuilding after a source change takes seconds.

## Step 2 — Inject the model

`inject_model.sh` does three things in sequence:
1. Runs `export_onnx.py` locally (CPU, no GPU needed) to convert the pruned weights to ONNX
2. Copies the resulting `model.onnx` into a temporary container with `docker cp`
3. Commits the container as a new image layer with `docker commit`

```bash
# From repo root, with venv active:
source .venv/bin/activate
bash vastai/inject_model.sh \
  dlshogi/experiments/exp032_structured_pruning/pruned_state_dict.pt \
  hmatsuya/dlshogi-usi:latest
```

Expected output:
```
=== Step 1: Export checkpoint → ONNX (CPU) ===
  checkpoint : dlshogi/experiments/exp032_structured_pruning/pruned_state_dict.pt
  output     : /tmp/dlshogi_model_<pid>.onnx
Device: cpu
Loading checkpoint: ...
Exporting to: /tmp/dlshogi_model_<pid>.onnx
Inlining external weights into single ONNX file...
  Final size: 262.8 MB
ONNX export complete.
ONNX model check passed.
  ONNX size  : 251M

=== Step 2: Inject model.onnx into image: hmatsuya/dlshogi-usi:latest ===
sha256:<new_image_id>

=== Done ===
  Image hmatsuya/dlshogi-usi:latest now contains /workspace/model/model.onnx

Next step — push to Docker Hub:
  docker push hmatsuya/dlshogi-usi:latest
```

### What export_onnx.py does

`export_onnx.py` accepts either a pruned state dict (`.pt`) or a Lightning checkpoint (`.ckpt`):

| Input format | Key structure | How it's loaded |
|---|---|---|
| `pruned_state_dict.pt` | `{"model": OrderedDict(...)}` | Unwraps the `"model"` key directly |
| Lightning `.ckpt` | `{"state_dict": {"model.<key>": ...}}` | Strips `"model."` / `"model._orig_mod."` prefix |

The exported ONNX has:
- **Inputs**: `input1` (board features, `[B, 62, 9, 9]`), `input2` (hand+flags, `[B, 57, 9, 9]`)
- **Outputs**: `output_policy` (`[B, 2187]`), `output_value` (`[B, 1]`, sigmoid applied)
- **Dynamic batch axis** on all four tensors
- **Opset 17**, weights inlined (single self-contained file, no `.data` sidecar)
- **Sigmoid on value**: the USI engine uses the ONNX value output directly as win probability in [0,1]. The sigmoid is applied inside the export wrapper so the raw logit is never exposed to the engine.

The model is `PrunedPolicyValueNetwork` from exp032 — InceptionNeXt with `mlp_expansion_dim=1536`
(25% MLP pruning from 2048), **65.7M parameters**.

### Updating the model

To swap in a new model without rebuilding the engine:

```bash
# 1. Inject new weights
source .venv/bin/activate
bash vastai/inject_model.sh <new_pruned_state_dict.pt> hmatsuya/dlshogi-usi:latest

# 2. Push
docker push hmatsuya/dlshogi-usi:latest
```

The engine binary and book are unchanged — only the model layer is replaced.

## Step 3 — Push to Docker Hub

```bash
docker push hmatsuya/dlshogi-usi:latest
```

## Step 4 — Create the Vast.ai template

```bash
bash vastai/create_template.sh
```

This prints the template hash. Save it — you'll use it to launch instances.

To update the template later (e.g. after pushing a new image):
```bash
TEMPLATE_HASH=<hash> bash vastai/create_template.sh
```

## Step 5 — Launch an instance on championship day

```bash
# Find a suitable GPU (RTX 4090 or better, CUDA >= 12.8, exclude China)
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cuda_max_good>=12.8 disk_space>=20 inet_down>=200' -o dph --raw \
  | python3 -c "
import sys, json
for o in json.load(sys.stdin):
    if ', CN' not in o.get('geolocation',''):
        print(f\"ID:{o['id']}  \${o['dph_total']:.3f}/hr  {o['geolocation']}\")
        break
"

# Create instance directly with the image (no template needed)
vastai create instance <OFFER_ID> --image hmatsuya/dlshogi-usi:latest --disk 20 --ssh --direct

# Get SSH connection details
vastai ssh-url <INSTANCE_ID>
```

## Step 6 — Connect and run

The ONNX model and Mafu opening book (ver11, Apery format) are already baked into the image — no upload needed.

```bash
# SSH in
ssh -p <PORT> root@<HOST>

# Run the engine (reads from stdin, USI protocol)
/workspace/run_usi.sh
```

## Step 7 — Connect ShogiHome (Windows) to the remote engine

The `vastai/` folder contains proxy scripts that make the remote USI engine
appear as a local engine to ShogiHome.

### Quick setup

1. **Copy** `usi_ssh_proxy.ps1` and `usi_ssh_proxy_ps.bat` to your Windows PC
   (same folder).

2. **Edit `usi_ssh_proxy.ps1`** — fill in the three variables at the top:

   ```powershell
   $VastHost = "123.45.67.89"      # Vast.ai instance public IP
   $VastPort = "12345"             # SSH port from the Vast.ai dashboard
   $KeyFile  = "C:\Users\you\.ssh\id_rsa"   # your private key
   ```

   Get the values from the Vast.ai dashboard or:
   ```bash
   vastai ssh-url <INSTANCE_ID>
   # prints: ssh://root@123.45.67.89:12345
   ```

3. **Register the engine in ShogiHome**:
   - Open ShogiHome → エンジン管理 (Engine Manager) → エンジン追加 (Add engine)
   - Select `usi_ssh_proxy_ps.bat`
   - ShogiHome calls the `.bat`, which calls the `.ps1`, which SSHes to the
     server and pipes USI commands through.

4. **Test the connection** before registering — open PowerShell and run:
   ```powershell
   .\usi_ssh_proxy.ps1
   # Type: usi
   # Expected response: id name dlshogi ...  usiok
   ```

### How it works

```
ShogiHome  ←stdin/stdout→  usi_ssh_proxy_ps.bat
                                    │
                            usi_ssh_proxy.ps1
                                    │
                            ssh.exe (OpenSSH)
                                    │  (TCP, port VAST_PORT)
                            Vast.ai server
                                    │
                            /workspace/run_usi.sh
                                    │
                            /usr/local/bin/usi  (TensorRT engine)
```

### Prerequisites (Windows)

- **OpenSSH client** — built-in on Windows 10/11.
  Check: `Settings > Apps > Optional Features > OpenSSH Client`
  Or install via PowerShell: `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0`
- **SSH key pair** — generate once with `ssh-keygen` and add the public key to
  the Vast.ai instance (paste into the instance's "SSH Keys" field in the
  Vast.ai web console, or use `ssh-copy-id`).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: ssh.exe not found` | Install OpenSSH client or Git for Windows |
| `Connection refused` | Check `VAST_PORT` — Vast.ai uses a non-standard port |
| Engine times out in ShogiHome | Firewall blocking the port; try `ssh -p PORT root@HOST echo ok` in CMD |
| `Permission denied (publickey)` | Wrong key file, or public key not added to the instance |
| First move very slow | TRT cache is being built on first run — normal, subsequent runs are fast |

---

## Notes

- **TRT serialized cache**: on first run the engine parses the ONNX and writes a `.serialized` file next to the model. Subsequent runs load the cache directly (much faster startup). The cache is GPU-architecture-specific — it will be regenerated if you switch GPU models.
- **Base image**: `nvcr.io/nvidia/tensorrt:25.03-py3` (TensorRT 10.8, CUDA 12.8, driver 570+). Supports RTX 4090 (Ada, sm_89) and RTX 5090 (Blackwell, sm_120). Requires host driver ≥ 570.
- **TRT workspace**: set to 4 GB in `usi/nn_tensorrt.cpp`. Required for the attention block (block 39) — TRT 10.4 cannot find a `Slice` kernel implementation with the default 64 MiB workspace.
- **ONNX opset**: 17 (legacy TorchScript exporter, IR v8). `LayerNormalization` is a first-class op in opset 17 and is natively supported by TRT 10.4. Do **not** downgrade to opset 16 or use the dynamo exporter (produces IR v10 which TRT cannot parse).
- **FP16**: enabled automatically if the GPU supports it (the engine checks `platformHasFastFp16()`).
- **Knowledge distillation checkpoints**: the loader in `export_onnx.py` strips `_teacher_model.*` keys automatically — KD checkpoints work the same as regular ones.
- **docker group**: `hmatsuya` is in the `docker` group — `sudo` is not needed for any `docker` commands.
- **Stopped vs destroyed instances**: `vastai stop` preserves the container filesystem (old image layers). Always use `vastai destroy` + `vastai create` when deploying a new image to ensure the fresh image is pulled.
- **Avoid China (CN) hosts**: Chinese Vast.ai hosts may have connectivity issues with Docker Hub and SSH latency from Japan. Filter them out with `', CN' not in geolocation` when selecting offers.
