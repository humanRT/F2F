"""Persistent, authenticated localhost SAM3 service. No images leave this PC."""
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

import numpy as np

from .progress import progress

HOST, PORT = "127.0.0.1", 49735
RUNTIME = Path(__file__).resolve().parents[2] / "F2F-results" / ".sam3-worker"
PROTOCOL = 1


def _token():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    path = RUNTIME / "token"
    try:
        with path.open("x") as f:
            f.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    return path.read_text().strip()


def _request(route, data=None, timeout=2):
    token = _token()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        body = json.dumps(data or {}).encode()
        conn.request("POST", route, body, {"Content-Type": "application/json", "X-F2F-Token": token})
        response = conn.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(result.get("error", f"Worker HTTP error {response.status}"))
        return result
    finally:
        conn.close()


def status():
    try:
        return _request("/status")
    except (OSError, http.client.HTTPException):
        return None


def ensure_worker():
    current = status()
    if current:
        if current.get("protocol") != PROTOCOL:
            raise RuntimeError("Worker version changed. Run: python main.py worker restart")
        return current
    main = Path(__file__).resolve().parents[1] / "main.py"
    with (RUNTIME / "worker.log").open("ab", buffering=0) as log:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen([sys.executable, "-B", str(main), "--sam3-worker"],
                                   cwd=RUNTIME, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   close_fds=True, creationflags=flags,
                                   start_new_session=os.name != "nt")
    progress("SAM3 worker", "Starting persistent local process...")
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        current = status()
        if current:
            return current
        if process.poll() is not None:
            # A concurrently started client may have won the bind race.
            current = status()
            if current:
                return current
            raise RuntimeError(f"Worker exited. See {RUNTIME / 'worker.log'}")
        time.sleep(.2)
    raise RuntimeError(f"Worker startup timed out. See {RUNTIME / 'worker.log'}")


def control(action):
    if action == "status":
        print(json.dumps(status() or {"running": False}, indent=2), flush=True)
        return
    if action in ("stop", "restart"):
        current = status()
        if current:
            _request("/stop")
            for _ in range(50):
                if status() is None:
                    break
                time.sleep(.1)
            else:
                raise RuntimeError("Worker is still stopping; try again shortly.")
        progress("SAM3 worker", "Stopped.")
    if action in ("start", "restart"):
        current = ensure_worker()
        progress("SAM3 worker", f"Ready (PID {current['pid']}); model loads on the first image.")


def predict(bgr, prompt, threshold=.5, checkpoint=None):
    import cv2
    current = ensure_worker()
    checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
    cached = current["model_loaded"] and current["checkpoint"] == checkpoint
    progress("SAM3 worker", f"PID {current['pid']}: " +
             ("using cached GPU model." if cached else "loading model for the first request/checkpoint."))
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Could not encode RGB for the SAM3 worker.")
    request = {"image": base64.b64encode(encoded).decode(), "prompt": prompt,
               "threshold": threshold, "checkpoint": checkpoint}
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_request, "/predict", request, 600)
        started = time.monotonic()
        while True:
            try:
                result = future.result(timeout=5)
                break
            except FutureTimeout:
                progress("SAM3 worker", f"Loading/inference in progress ({time.monotonic()-started:.0f}s)...")
    with np.load(io.BytesIO(base64.b64decode(result["arrays"])), allow_pickle=False) as data:
        masks, scores = data["masks"].copy(), data["scores"].copy()
    progress("SAM3 worker", f"Finished in {result['seconds']:.2f}s; {len(masks)} instance(s); "
             f"model {'reused' if result['reused'] else 'loaded'}.")
    return masks, scores


def serve():
    import cv2
    from .reused_3dmasks import Segmenter
    token = _token()
    lock = threading.Lock()
    state = {"segmenter": None, "checkpoint": None, "requests": 0, "busy": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("X-F2F-Token", ""), token):
                self.send_json(403, {"error": "F2F worker authentication failed."})
                return
            if self.path == "/status":
                model = state["segmenter"]
                self.send_json(200, {"running": True, "protocol": PROTOCOL, "pid": os.getpid(),
                                    "model_loaded": model is not None and model.processor is not None,
                                    "checkpoint": state["checkpoint"], "requests": state["requests"],
                                    "busy": state["busy"]})
                return
            if self.path == "/stop":
                if not lock.acquire(blocking=False):
                    self.send_json(409, {"error": "Worker is busy. Wait for inference before stopping it."})
                    return
                try:
                    self.send_json(200, {"stopping": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                finally:
                    lock.release()
                return
            if self.path != "/predict":
                self.send_json(404, {"error": "Unknown endpoint."})
                return
            if not lock.acquire(blocking=False):
                self.send_json(409, {"error": "SAM3 is processing another request. Try again when it finishes."})
                return
            state["busy"] = True
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 64*1024*1024:
                    raise ValueError("Invalid image request size.")
                data = json.loads(self.rfile.read(length))
                threshold = float(data.get("threshold", .5))
                if not 0 <= threshold <= 1:
                    raise ValueError("Threshold must be between 0 and 1.")
                bgr = cv2.imdecode(np.frombuffer(base64.b64decode(data["image"], validate=True),
                                                dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or bgr.size > 30_000_000:
                    raise ValueError("Invalid or oversized image.")
                checkpoint = data.get("checkpoint")
                reused = (state["segmenter"] is not None and state["checkpoint"] == checkpoint
                          and state["segmenter"].processor is not None)
                if not reused:
                    state["segmenter"] = None
                    import gc
                    gc.collect()
                    if "torch" in sys.modules:
                        sys.modules["torch"].cuda.empty_cache()
                    state["segmenter"] = Segmenter(checkpoint)
                    state["checkpoint"] = checkpoint
                started = time.monotonic()
                masks, scores = state["segmenter"].predict(bgr, data["prompt"], threshold)
                state["requests"] += 1
                buffer = io.BytesIO()
                np.savez_compressed(buffer, masks=masks, scores=scores)
                self.send_json(200, {"arrays": base64.b64encode(buffer.getvalue()).decode(),
                                     "seconds": time.monotonic()-started, "reused": reused})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            finally:
                state["busy"] = False
                lock.release()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    progress("SAM3 worker", f"Listening locally on {HOST}:{PORT}, PID {os.getpid()}")
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
