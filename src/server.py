# Dipendenze: higgs_loader.py, llama_engine/llama_higgs_engine.py (predefinito) o fast_engine.py (--engine torch),
# text_chunks.py, audio_tidy.py. Avvio: avvia_server.bat / higgs_client.py.
# Server vocale Higgs v3 senza ComfyUI. POST /tts {"text","voice",...} -> WAV; GET /health; GET /voices.
# Ogni testo e' spezzato in frasi; le frasi di tutte le richieste in coda vengono generate insieme.
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_engine import FastHiggsEngine
from higgs_loader import VOICES_DIR, encode_voice, load_bundle
from audio_tidy import tidy_chunk
from text_chunks import split_text


class ChunkJob:
    def __init__(self, text, voice, params):
        self.text, self.voice, self.params = text, voice, params
        self.audio = None
        self.error = None
        self.done = threading.Event()


class SpeechWorker:
    def __init__(self, args):
        self.args = args
        self.bundle = load_bundle(codec_device=args.codec_device)
        if args.engine == "llama":
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_engine"))
            from llama_higgs_engine import LlamaHiggsEngine  # cervello in llama.cpp: ~15 volte piu' veloce
            self.engine = LlamaHiggsEngine(self.bundle, args.batch, args.max_cache_len)
        else:
            self.engine = FastHiggsEngine(self.bundle, args.batch, args.max_cache_len, use_graph=False)
        self.voice_cache = {}
        self.jobs: queue.Queue[ChunkJob] = queue.Queue()
        self.lock = threading.Lock()
        self.active_requests = 0
        self.last_activity = time.monotonic()
        threading.Thread(target=self._loop, daemon=True).start()

    def voice(self, name):
        if name not in self.voice_cache:
            with torch.inference_mode():
                self.voice_cache[name] = encode_voice(self.bundle, name)
        return self.voice_cache[name]

    def _collect_batch(self):
        batch = [self.jobs.get()]
        deadline = time.monotonic() + self.args.gather_seconds
        while len(batch) < self.args.batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self.jobs.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _loop(self):
        while True:
            batch = self._collect_batch()
            groups = {}
            for job in batch:  # one generate per sampling setting
                key = (job.params["temperature"], job.params["top_p"], job.params["top_k"], job.params["max_new_tokens"])
                groups.setdefault(key, []).append(job)
            for (temperature, top_p, top_k, max_new_tokens), jobs in groups.items():
                try:
                    self._run(jobs, temperature, top_p, top_k, max_new_tokens)
                except Exception as error:  # report to every waiting request
                    for job in jobs:
                        job.error = str(error)
                finally:
                    for job in jobs:
                        job.done.set()

    def _run(self, jobs, temperature, top_p, top_k, max_new_tokens):
        native = self.bundle.native
        requests = []
        for job in jobs:
            ref_delayed, ref_text = self.voice(job.voice)
            ids = self.bundle.tokenizer.build_prompt(job.text, num_ref_tokens=int(ref_delayed.shape[0]),
                                                     reference_text=ref_text or None)
            requests.append((ids, ref_delayed))
        seed = jobs[0].params.get("seed")
        if seed:
            torch.manual_seed(int(seed))
        started = time.perf_counter()
        delayed_list = self.engine.generate(requests, max_new_tokens=max_new_tokens, temperature=temperature,
                                            top_p=None if top_p <= 0 or top_p >= 1 else top_p,
                                            top_k=None if top_k <= 0 else top_k)
        generated_s = time.perf_counter() - started
        print(f"batch={len(jobs)} steps={self.engine.last_steps} rows={[d.shape[0] for d in delayed_list]} "
              f"gen_s={generated_s:.1f} ms_per_step={1000 * generated_s / max(self.engine.last_steps, 1):.0f}", flush=True)
        # one codec call per chunk: codec time grows much faster than length (750 frames 0.2 s, 2240 frames 11 s)
        started = time.perf_counter()
        with torch.inference_mode():
            for job, delayed in zip(jobs, delayed_list):
                if delayed.shape[0] < self.bundle.model.num_codebooks:
                    raise RuntimeError("too few audio rows generated")
                job.audio = self.bundle.codec.decode(native.reverse_delay_pattern(delayed)).numpy()
        print(f"decode_s={time.perf_counter() - started:.1f}", flush=True)


def make_handler(state: dict, args):
    """state["worker"] is None while the model is loading: /health answers at once (ready=false), /tts says 503."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, payload):
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

        def do_GET(self):
            worker = state["worker"]
            if self.path == "/health":
                if worker is None:
                    return self._json(200, {"ok": True, "ready": False, "loading": True, "busy": False})
                self._json(200, {"ok": True, "ready": True, "loading": False,
                                 "busy": worker.active_requests > 0 or not worker.jobs.empty(),
                                 "active_requests": worker.active_requests, "queued_chunks": worker.jobs.qsize(),
                                 "batch": args.batch})
            elif self.path == "/voices":
                self._json(200, sorted(p.stem for p in VOICES_DIR.glob("*.wav")))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/tts":
                return self._json(404, {"error": "not found"})
            worker = state["worker"]
            if worker is None:
                return self._json(503, {"error": "il modello si sta ancora caricando: riprova fra poco"})
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                text, voice = request["text"].strip(), request["voice"]
                if not (VOICES_DIR / f"{voice}.wav").is_file():
                    return self._json(400, {"error": f"voice not found: {voice}"})
                params = {
                    "temperature": float(request.get("temperature", 1.0)),
                    "top_p": float(request.get("top_p", 0.95)),
                    "top_k": int(request.get("top_k", 0)),
                    "max_new_tokens": int(request.get("max_new_tokens", args.max_new_tokens)),
                    "seed": request.get("seed"),
                }
                chunks = split_text(text, int(request.get("max_chars", args.max_chars)),
                                    int(request.get("min_chars", args.min_chars)))
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                return self._json(400, {"error": f"bad request: {error}"})
            started = time.perf_counter()
            jobs = [ChunkJob(chunk, voice, params) for chunk in chunks]
            with worker.lock:
                worker.active_requests += 1
            try:
                for job in jobs:
                    worker.jobs.put(job)
                for job in jobs:
                    job.done.wait()
            finally:
                with worker.lock:
                    worker.active_requests -= 1
                    worker.last_activity = time.monotonic()
            errors = [job.error for job in jobs if job.error]
            if errors:
                return self._json(500, {"error": errors[0]})
            sample_rate = worker.bundle.native.SAMPLE_RATE
            pause = np.zeros(int(sample_rate * float(request.get("pause_seconds", args.pause_seconds))), dtype=np.float32)
            pieces = []
            for index, job in enumerate(jobs):
                if index:
                    pieces.append(pause)
                cleaned, warning = tidy_chunk(job.audio.astype(np.float32), sample_rate)
                if warning:
                    self.log_message("pezzo %d non ripulito: %s", index, warning)
                pieces.append(cleaned)
            buffer = io.BytesIO()
            soundfile.write(buffer, np.concatenate(pieces), sample_rate, format="WAV")
            self.log_message("tts chunks=%d seconds=%.1f", len(jobs), time.perf_counter() - started)
            self._send(200, buffer.getvalue(), "audio/wav")

    return Handler


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """On Windows the standard server may bind a port already in use (address reuse): here the port is exclusive."""
    allow_reuse_address = False

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _shutdown_when_idle(server, worker, idle_seconds):
    """Stops the process (freeing the video memory) after idle_seconds with no request in progress."""
    while True:
        time.sleep(5)
        busy = worker.active_requests > 0 or not worker.jobs.empty()
        if busy:
            worker.last_activity = time.monotonic()
        elif time.monotonic() - worker.last_activity > idle_seconds:
            print(f"nessuna richiesta da {idle_seconds} s: spengo il server", flush=True)
            server.shutdown()
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--batch", type=int, default=12, help="frasi generate insieme")
    parser.add_argument("--max-cache-len", type=int, default=8192,  # = contesto massimo del modello; la memoria si prende per richiesta
                        help="posizioni massime per testo (campione voce + testo + voce generata)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--codec-device", default="cuda", help="sempre la scheda video (regola dell'utente)")
    parser.add_argument("--engine", choices=("llama", "torch"), default="llama",
                        help="llama = cervello in llama.cpp (veloce); torch = motore precedente")
    parser.add_argument("--idle-shutdown", type=float, default=120, help="secondi senza lavoro prima di spegnersi (0 = mai)")
    parser.add_argument("--gather-seconds", type=float, default=0.3)
    args = parser.parse_args()
    state = {"worker": None}
    try:  # la porta si prenota PRIMA del modello: un secondo avvio si ferma qui senza toccare la scheda
        server = ExclusiveHTTPServer(("127.0.0.1", args.port), make_handler(state, args))
    except OSError:
        print(f"Porta {args.port} gia' occupata: c'e' gia' un server voce acceso, non ne avvio un secondo.", flush=True)
        sys.exit(3)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    state["worker"] = worker = SpeechWorker(args)
    if args.idle_shutdown > 0:
        threading.Thread(target=_shutdown_when_idle, args=(server, worker, args.idle_shutdown), daemon=True).start()
    print(f"Higgs server pronto su http://127.0.0.1:{args.port} (batch {args.batch})", flush=True)
    serving.join()


if __name__ == "__main__":
    main()
