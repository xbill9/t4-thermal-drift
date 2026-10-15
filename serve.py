"""One vLLM server on the T4 with a gemma4-dev rig's serving flags (its tpu.env, as ~/bin/vllm-t4
passes them), optionally with a Gemma 4 drafter. Copied from
gemma4-dev/gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-10-15-overnight-t4/t4serve.py and cut down.
"""
import json, os, signal, subprocess, time, urllib.request

RIG = os.environ.get("RIG", os.path.expanduser("~/gemma4-dev/gpu-vllm-t4-2b-w4a16"))


def load_env(rig=RIG):
    env = {}
    for line in open(os.path.join(rig, "tpu.env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip('"')
    return env


def start(env, logpath, draft=None, n=None):
    cmd = [env["PYTHON_BIN"], "-m", "vllm.entrypoints.openai.api_server", "--model", env["MODEL_NAME"],
           "--host", env["HOST"], "--port", env["PORT"], "--dtype", env["DTYPE"],
           "--kv-cache-dtype", env["KV_CACHE_DTYPE"], "--tensor-parallel-size", env["TENSOR_PARALLEL_SIZE"],
           "--gpu-memory-utilization", env["GPU_MEMORY_UTILIZATION"], "--max-model-len", env["MAX_MODEL_LEN"],
           "--max-num-seqs", env["MAX_NUM_SEQS"]]
    if env.get("LANGUAGE_MODEL_ONLY") == "true":
        cmd.append("--language-model-only")
    if n:
        cmd += ["--speculative-config", json.dumps({"model": draft, "num_speculative_tokens": n})]
    senv = dict(os.environ, PYTHONUSERBASE=env["PYTHONUSERBASE"], TMPDIR=env["TMPDIR"], HF_HUB_OFFLINE="1")
    log = open(logpath, "w")
    log.write(" ".join(cmd) + "\n"); log.flush()
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=senv, start_new_session=True)
    while True:
        try:
            urllib.request.urlopen(env["ENDPOINT"] + "/health", timeout=2); break
        except Exception:
            if srv.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup, see {logpath}")
            if time.time() - t0 > 1800:
                stop(srv, log); raise RuntimeError(f"not healthy after 1800 s, see {logpath}")
            time.sleep(5)
    return srv, log


def stop(srv, log):
    try:
        os.killpg(srv.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    srv.wait(); log.close(); time.sleep(10)


def stream_chat(url, model, body, timeout=900):
    """One streamed chat completion. Returns text, completion tokens, request start, and the wall
    time (time.time()) of every streamed content chunk."""
    body = dict(body, model=model, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); times, text, n = [], [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and d["choices"][0]["delta"].get("content"):
                times.append(time.time()); text.append(d["choices"][0]["delta"]["content"])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    return "".join(text), n, t0, times
