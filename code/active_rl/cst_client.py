"""CST TCP client (ported from v6/v8) and world-model mock client.

JSON + 4-byte big-endian length-prefix framing over TCP, compatible with the
CST v3 server (default port 65433).
"""

import json
import socket
import time

import numpy as np
import torch

from active_rl.config import DEVICE


def recvall(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("CST connection closed")
        data += chunk
    return data

class CSTClient:
    """CST v3 client: 10x10 pattern + theta + TE/TM/BOTH."""
    def __init__(self, host="127.0.0.1", port=65433, timeout=300):
        self.host = host; self.port = port; self.timeout = timeout
        self.sock = None; self.cache = {}; self.n_calls = 0
        self._connect()

    def _connect(self):
        if self.sock:
            try: self.sock.close()
            except: pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print(f"  CST v3 connected: {self.host}:{self.port}")

    def _request(self, payload_dict):
        payload = json.dumps(payload_dict).encode()
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                self.sock.sendall(len(payload).to_bytes(4, "big"))
                self.sock.sendall(payload)
                header = recvall(self.sock, 4)
                n = int.from_bytes(header, "big")
                resp = json.loads(recvall(self.sock, n).decode())
                if "error" in resp:
                    raise RuntimeError(f"CST server error: {resp['error']}")
                return resp
            except (ConnectionResetError, ConnectionAbortedError,
                    ConnectionError, TimeoutError, OSError) as e:
                if attempt < max_retries:
                    print(f"  CST retry {attempt+1}/{max_retries}: {e}")
                    time.sleep(2)
                    self._connect()
                else:
                    raise

    def _parse_response_pair(self, resp):
        s11 = np.abs(np.array(resp["s11_real"], dtype=np.float32) +
                     1j * np.array(resp["s11_imag"], dtype=np.float32))
        s21 = np.abs(np.array(resp["s21_real"], dtype=np.float32) +
                     1j * np.array(resp["s21_imag"], dtype=np.float32))
        return s11, s21

    def simulate(self, pattern_10x10, theta=0.0, pol="TE"):
        pol_str = str(pol) if isinstance(pol, str) else ("TE" if pol == 0 else "TM")
        key = (tuple(pattern_10x10.flatten().astype(int).tolist()), theta, pol_str)
        if key in self.cache:
            return self.cache[key]

        self.n_calls += 1
        resp = self._request({
            "pattern": pattern_10x10.astype(int).tolist(),
            "theta": float(theta),
            "pol": pol_str,
        })
        s11, s21 = self._parse_response_pair(resp)
        self.cache[key] = (s11, s21)
        return s11, s21

    def simulate_both(self, pattern_10x10, theta=0.0):
        base_key = tuple(pattern_10x10.flatten().astype(int).tolist())
        key_te = (base_key, theta, "TE")
        key_tm = (base_key, theta, "TM")
        if key_te in self.cache and key_tm in self.cache:
            return {0: self.cache[key_te], 1: self.cache[key_tm]}

        self.n_calls += 1
        resp = self._request({
            "pattern": pattern_10x10.astype(int).tolist(),
            "theta": float(theta),
            "pol": "BOTH",
        })
        te = self._parse_response_pair(resp["TE"])
        tm = self._parse_response_pair(resp["TM"])
        self.cache[key_te] = te
        self.cache[key_tm] = tm
        return {0: te, 1: tm}

    def close(self):
        try: self.sock.close()
        except: pass


class MockCST:
    """Mock CST v3: return world-model predictions for requested polarization."""
    def __init__(self, wm): self.wm = wm; self.n_calls = 0; self.cache = {}

    def simulate(self, pattern_10x10, theta=0.0, pol="TE"):
        self.n_calls += 1
        pol_int = 0 if str(pol) in ("TE", "0") else 1
        M = torch.from_numpy(pattern_10x10).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
        s11, s21 = self.wm.predict(M, theta=theta, pol=pol_int)
        return s11.cpu().numpy().flatten(), s21.cpu().numpy().flatten()

    def simulate_both(self, pattern_10x10, theta=0.0):
        self.n_calls += 1
        M = torch.from_numpy(pattern_10x10).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
        te_s11, te_s21 = self.wm.predict(M, theta=theta, pol=0)
        tm_s11, tm_s21 = self.wm.predict(M, theta=theta, pol=1)
        return {
            0: (te_s11.cpu().numpy().flatten(), te_s21.cpu().numpy().flatten()),
            1: (tm_s11.cpu().numpy().flatten(), tm_s21.cpu().numpy().flatten()),
        }

    def close(self): pass
