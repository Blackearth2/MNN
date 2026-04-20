# Neural Network Trainer

A custom neural network for image classification (MNIST), with:
- **C backend** — fast training with your own architecture
- **Python server** — Flask + SocketIO bridge
- **Web UI** — real-time loss/accuracy charts, network diagram, live log

---

## Setup

### 1. Install Python dependencies
```bash
pip install flask flask-socketio eventlet numpy
```

### 2. Compile the C trainer
```bash
make
```
This produces `nn_train` — the binary that does all the math.

### 3. Start the server
```bash
python server.py
```

### 4. Open the UI
```
http://localhost:5000
```

---

## Using the UI

| Control | Description |
|---|---|
| **Training samples** | How many MNIST images to use (100–60 000). More = slower but better accuracy. |
| **Epochs** | Full passes through the dataset. |
| **Learning rate** | Slider from 0.0001 to 0.1 (log scale). |
| **Report every N** | How often the C process emits a step-level progress event. |
| **Architecture** | Add/remove hidden layers. Each has a neuron count + activation. |
| **▶ Train** | Starts training. MNIST is auto-downloaded (~12 MB) on first run. |
| **■ Stop** | Gracefully terminates the C process. |

---

## Architecture

```
data/                  ← MNIST binary files (auto-downloaded)
c/
  layer.h / layer.c    ← Layer, forward/backprop, activations, gradient descent
  network.h / network.c← Network, training loop (outputs JSON lines)
  main.c               ← CLI entry point, binary data loader
nn_train               ← compiled binary
server.py              ← Flask + SocketIO server
templates/index.html   ← Single-file frontend
Makefile
```

### Binary data format written by Python, read by C
```
int32   num_samples
int32   input_size       (784 for MNIST)
int32   output_size      (10 for digits)
float32 [num_samples × input_size]   pixel values, normalised 0–1
float32 [num_samples × output_size]  one-hot labels
```

### JSON events emitted by C → Python → WebSocket → browser
```json
{"type":"config",   "layers":[784,128,64,10], "epochs":10, "samples":5000}
{"type":"progress", "epoch":1, "step":200, "total":5000, "loss":1.2345}
{"type":"epoch",    "epoch":1, "total_epochs":10, "loss":0.8, "accuracy":0.75}
{"type":"done"}
```

---

## Bugs fixed from original code

| Bug | Fix |
|---|---|
| Output layer `delta[]` never set → corrupted hidden-layer backprop | Set `delta[i]` in output layer loop |
| Magic number `9` in `compute_mse_loss` call | Use actual `output_size` |
| Uniform random weight init → slow convergence | Xavier uniform init |
| `get_activation_function` crashes on LINEAR / SOFTMAX | Returns `linear_f` for both |
| Softmax missing from forward pass | `softmax_inplace()` applied after linear pre-activation |

---

## Extending

- **New dataset**: Write your own `prepare_dataset()` in `server.py` that produces the same binary format.
- **New optimizer**: Add a function in `layer.c` following the pattern of `batchGradientDescent`.
- **Save/load weights**: Add `fwrite`/`fread` on `layer->weights` + `layer->biases` in a new `saveNetwork` / `loadNetwork` function.
