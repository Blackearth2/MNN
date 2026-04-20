"""
server.py  –  Neural Network Trainer + Inference
Run:  python server.py  →  http://localhost:5000

Custom datasets: drop CSV files in data/custom/
  Last column = integer class label (0-based). Header optional.
"""
import os, struct, json, subprocess, gzip, urllib.request
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CUSTOM_DIR = os.path.join(DATA_DIR, "custom")
BIN_EXEC   = os.path.join(BASE_DIR, "nn_train")
MODEL_PATH = os.path.join(BASE_DIR, "last_model.bin")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(CUSTOM_DIR, exist_ok=True)

_rdm = os.path.join(CUSTOM_DIR,"README.txt")
if not os.path.exists(_rdm):
    open(_rdm,"w").write(
        "Drop CSV files here.\nLast column = integer class (0-based). Header optional.\n"
        "Features are auto-normalised to [0,1].\n")

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "nn-trainer"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

training_process = None
stop_requested   = False

# ── Loaded model (for inference) ─────────────────────────────────────────────
loaded_model = {
    "ready":       False,
    "dataset":     None,    # "mnist" or csv filename
    "input_size":  0,
    "output_size": 0,
    "layer_sizes": [],      # all layer output sizes
    "act_types":   [],      # per layer (0=RELU,1=SIG,2=TANH,3=LIN,4=SOFT)
    "weights":     [],      # list of np [out, in]
    "biases":      [],      # list of np [out]
}

def load_model(path):
    """Parse NNW2 binary and populate loaded_model."""
    try:
        with open(path,"rb") as f:
            magic = f.read(4)
            if magic not in (b"NNW2", b"NNWT"):
                raise ValueError(f"bad magic: {magic}")
            nb, = struct.unpack("<i", f.read(4))

            if magic == b"NNW2":
                # v2: explicit input_sizes + output_sizes
                in_sizes  = list(struct.unpack(f"<{nb}i", f.read(4*nb)))
                out_sizes = list(struct.unpack(f"<{nb}i", f.read(4*nb)))
                acts      = list(struct.unpack(f"<{nb}i", f.read(4*nb)))
            else:
                # v1 legacy (broken square matrices — refuse gracefully)
                raise ValueError("Old model format (NNWT) detected — please retrain.")

            weights, biases = [], []
            for i in range(1, nb):
                isz, osz = in_sizes[i], out_sizes[i]
                expected_bytes = 4 * isz * osz
                raw_w = f.read(expected_bytes)
                if len(raw_w) != expected_bytes:
                    raise ValueError(
                        f"Layer {i}: expected {expected_bytes} bytes for weights "
                        f"({osz}×{isz}), got {len(raw_w)}. "
                        f"Model file may be corrupt or from an older version.")
                w = np.frombuffer(raw_w, dtype=np.float32).reshape(osz, isz).copy()

                raw_b = f.read(4 * osz)
                if len(raw_b) != 4 * osz:
                    raise ValueError(f"Layer {i}: truncated bias data")
                b = np.frombuffer(raw_b, dtype=np.float32).copy()

                weights.append(w)
                biases.append(b)

        layer_sizes = out_sizes  # UI shows output size per layer
        loaded_model.update({
            "ready":       True,
            "layer_sizes": layer_sizes,
            "act_types":   acts,
            "input_size":  in_sizes[0],
            "output_size": out_sizes[-1],
            "weights":     weights,
            "biases":      biases,
        })
        # Store flat weights for neuron inspector
        import builtins
        builtins.__dict__['_modelWeights'] = [w.flatten().tolist() for w in weights]

        socketio.emit("model_ready", {
            "input_size":  in_sizes[0],
            "output_size": out_sizes[-1],
            "layer_sizes": layer_sizes,
            "dataset":     loaded_model["dataset"],
        })
        print(f"  Model loaded: {' → '.join(str(s) for s in layer_sizes)}", flush=True)
    except Exception as e:
        msg = f"Failed to load model: {e}"
        print(msg, flush=True)
        socketio.emit("log", {"msg": msg, "cls": "danger"})

def _relu(x):    return np.maximum(0, x)
def _sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
def _softmax(x):
    e = np.exp(x - x.max()); return e/e.sum()

ACT_FNS = [_relu, _sigmoid, np.tanh, lambda x: x, _softmax]

def run_inference(pixel_array):
    """pixel_array: list of floats, length = input_size.
       Returns {probabilities, predicted, activations_per_layer}"""
    m = loaded_model
    if not m["ready"]: return None
    x = np.array(pixel_array, dtype=np.float32)
    layer_acts = [x.tolist()]
    for w, b, at in zip(m["weights"], m["biases"], m["act_types"][1:]):
        x = w @ x + b
        x = ACT_FNS[at](x)
        layer_acts.append(x.tolist())
    return {
        "probabilities": layer_acts[-1],
        "predicted":     int(np.argmax(layer_acts[-1])),
        "activations":   layer_acts,   # one array per layer incl. input
    }

# ── MNIST ─────────────────────────────────────────────────────────────────────
MNIST_URLS = {
    "train-images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}

def download_mnist():
    for name, url in MNIST_URLS.items():
        raw = os.path.join(DATA_DIR, name+".bin")
        if os.path.exists(raw): continue
        emit_safe("log",{"msg":f"Downloading {name}…","cls":"amber"})
        dest = raw+".gz"
        urllib.request.urlretrieve(url, dest)
        with gzip.open(dest,"rb") as gz, open(raw,"wb") as out: out.write(gz.read())
        os.remove(dest)
        emit_safe("log",{"msg":f"{name} ready.","cls":""})

def load_mnist_images(path):
    with open(path,"rb") as f:
        f.read(16); data=np.frombuffer(f.read(),dtype=np.uint8)
    return data.reshape(-1,784).astype(np.float32)/255.0

def load_mnist_labels(path):
    with open(path,"rb") as f:
        f.read(8); labels=np.frombuffer(f.read(),dtype=np.uint8)
    one_hot=np.zeros((len(labels),10),dtype=np.float32)
    one_hot[np.arange(len(labels)),labels]=1.0
    return one_hot

def prepare_mnist(subset):
    bp=os.path.join(DATA_DIR,f"mnist_{subset}.bin")
    if os.path.exists(bp): return bp,784,10
    download_mnist()
    imgs=load_mnist_images(os.path.join(DATA_DIR,"train-images.bin"))
    lbls=load_mnist_labels(os.path.join(DATA_DIR,"train-labels.bin"))
    idx=np.random.permutation(len(imgs))[:subset]
    write_binary(bp,imgs[idx],lbls[idx])
    return bp,784,10

# ── Custom CSV ────────────────────────────────────────────────────────────────
def prepare_custom(filename, subset):
    csv_path=os.path.join(CUSTOM_DIR,filename)
    if not os.path.exists(csv_path): raise FileNotFoundError(csv_path)
    emit_safe("log",{"msg":f"Loading {filename}…","cls":""})
    rows=[]
    with open(csv_path) as f:
        for line in f:
            line=line.strip()
            if line: rows.append(line.split(","))
    try: float(rows[0][-1])
    except ValueError: rows=rows[1:]
    data=np.array(rows,dtype=np.float32)
    labels=data[:,-1].astype(int); feats=data[:,:-1]
    mn,mx=feats.min(0),feats.max(0)
    feats=(feats-mn)/np.where(mx-mn==0,1,mx-mn)
    nc=int(labels.max())+1
    oh=np.zeros((len(labels),nc),dtype=np.float32)
    oh[np.arange(len(labels)),labels]=1.0
    if subset and subset<len(feats):
        idx=np.random.permutation(len(feats))[:subset]
        feats=feats[idx]; oh=oh[idx]
    bp=os.path.join(DATA_DIR,f"custom_{filename}_{len(feats)}.bin")
    write_binary(bp,feats,oh)
    emit_safe("log",{"msg":f"{len(feats)} samples · {feats.shape[1]} features · {nc} classes","cls":"accent"})
    return bp,feats.shape[1],nc

def write_binary(path,images,labels):
    n,isz=images.shape; _,osz=labels.shape
    with open(path,"wb") as f:
        f.write(struct.pack("<iii",n,isz,osz))
        f.write(images.tobytes()); f.write(labels.tobytes())

# ── Build ─────────────────────────────────────────────────────────────────────
def build_c():
    r=subprocess.run(["make","-C",BASE_DIR],capture_output=True,text=True)
    if r.returncode!=0: raise RuntimeError(f"make failed:\n{r.stderr}")

def emit_safe(event, data):
    socketio.emit(event, data)

# ── Training task ─────────────────────────────────────────────────────────────
def run_training(config):
    global training_process, stop_requested
    stop_requested=False
    try:
        emit_safe("status",{"state":"preparing"})
        emit_safe("log",{"msg":"Building C binary…","cls":""})
        build_c()

        dataset=config.get("dataset","mnist")
        subset =int(config.get("num_samples",5000) or 5000)

        if dataset=="mnist":
            emit_safe("log",{"msg":"Preparing MNIST…","cls":""})
            data_path,input_size,output_size=prepare_mnist(subset)
        else:
            data_path,input_size,output_size=prepare_custom(dataset,subset)

        loaded_model["dataset"]=dataset

        hidden=config.get("hidden_layers",[{"size":128,"activation":0}])
        acts  =[3]+[int(h["activation"]) for h in hidden]+[4]
        sizes =[input_size]+[int(h["size"]) for h in hidden]+[output_size]

        cmd=[BIN_EXEC, data_path,
             str(int(config.get("epochs",10))),
             str(float(config.get("lr",0.01))),
             str(int(config.get("report_every",200))),
             MODEL_PATH,
             *[str(a) for a in acts],
             *[str(s) for s in sizes]]

        emit_safe("log",{"msg":f"Layers: {' → '.join(str(s) for s in sizes)}","cls":"blue"})
        emit_safe("status",{"state":"training"})

        training_process=subprocess.Popen(
            cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)

        for line in training_process.stdout:
            if stop_requested: training_process.terminate(); break
            line=line.strip()
            if not line: continue
            try:
                msg=json.loads(line)
                emit_safe("train_event",msg)
                if msg.get("type")=="saved":
                    load_model(msg["path"])
            except json.JSONDecodeError:
                emit_safe("log",{"msg":line,"cls":""})

        training_process.wait()
        stderr=training_process.stderr.read().strip()
        if stderr:
            for l in stderr.splitlines()[:10]:
                emit_safe("log",{"msg":l,"cls":"amber"})

        emit_safe("status",{"state":"stopped" if stop_requested else "done"})
    except Exception as e:
        emit_safe("status",{"state":"error"})
        emit_safe("log",{"msg":f"ERROR: {e}","cls":"danger"})
    finally:
        training_process=None

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/datasets")
def api_datasets():
    files=[f for f in os.listdir(CUSTOM_DIR) if f.endswith(".csv")]
    return jsonify({"files":sorted(files)})

@app.route("/api/start",methods=["POST"])
def api_start():
    global training_process
    if training_process: return jsonify({"error":"already running"}),400
    socketio.start_background_task(run_training, request.json or {})
    return jsonify({"status":"started"})

@app.route("/api/stop",methods=["POST"])
def api_stop():
    global stop_requested,training_process
    stop_requested=True
    if training_process: training_process.terminate()
    return jsonify({"status":"stopping"})

@app.route("/api/predict",methods=["POST"])
def api_predict():
    if not loaded_model["ready"]:
        return jsonify({"error":"no model loaded"}),400
    data=request.json or {}
    pixels=data.get("pixels",[])
    if len(pixels)!=loaded_model["input_size"]:
        return jsonify({"error":f"expected {loaded_model['input_size']} values, got {len(pixels)}"}),400
    result=run_inference(pixels)
    return jsonify(result)

@socketio.on("connect")
def on_connect():
    emit("status",{"state":"idle"})
    files=[f for f in os.listdir(CUSTOM_DIR) if f.endswith(".csv")]
    emit("datasets",{"files":sorted(files)})
    if loaded_model["ready"]:
        emit("model_ready",{
            "input_size":  loaded_model["input_size"],
            "output_size": loaded_model["output_size"],
            "layer_sizes": loaded_model["layer_sizes"],
            "dataset":     loaded_model["dataset"],
        })

if __name__=="__main__":
    print(f"\n  Neural Network Trainer → http://localhost:5000")
    print(f"  Custom datasets        → {CUSTOM_DIR}\n")
    socketio.run(app,host="0.0.0.0",port=5000,debug=False,allow_unsafe_werkzeug=True)
