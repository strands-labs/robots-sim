#!/usr/bin/env python3
"""GR00T inference client — thin ZMQ wrapper for policy server communication.

Supports two transport protocols controlled by the ``protocol`` parameter:

* ``"sim_wrapper"`` — Wraps observations in ``{"observation": obs}`` and
  handles tuple ``(action, info)`` responses.  Used by Isaac-GR00T N1.6+
  servers running ``Gr00tSimPolicyWrapper``.

* ``"direct"`` — Sends observations as a flat data dict.  Used by
  Isaac-GR00T N1.5 servers.

* ``"auto"`` (default) — Tries ``sim_wrapper`` first, falls back to
  ``direct`` on error.

SPDX-License-Identifier: Apache-2.0
"""

import io

import msgpack
import numpy as np
import zmq


def _encode(obj):
    """Encode numpy arrays for msgpack transport."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    """Decode numpy arrays from msgpack transport."""
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class GR00TClient:
    """Minimal ZMQ client for GR00T inference servers."""

    def __init__(self, host="localhost", port=5555, protocol="auto"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://{host}:{port}")
        self.protocol = protocol

    def get_action(self, observations):
        """Send observations and receive an action chunk.

        The request format adapts to the configured protocol.
        """
        if self.protocol == "sim_wrapper":
            return self._request({"observation": observations})
        elif self.protocol == "direct":
            return self._request_flat(observations)
        else:
            # Auto-detect: try sim_wrapper first; if the response lacks
            # action keys, retry with the direct protocol.
            try:
                result = self._request({"observation": observations})
                if self._has_action_keys(result):
                    return result
            except Exception:
                pass
            return self._request_flat(observations)

    def _request(self, data):
        """Send wrapped request and parse response."""
        raw = self._send_recv(data)
        # sim_wrapper returns (action_dict, info_dict) tuple
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return raw[0]
        return raw

    def _request_flat(self, observations):
        """Send flat request (legacy protocol)."""
        return self._send_recv(observations)

    def _send_recv(self, data):
        """Low-level send/receive with error handling."""
        request = {"endpoint": "get_action", "data": data}
        self.sock.send(msgpack.packb(request, default=_encode))
        response = msgpack.unpackb(self.sock.recv(), object_hook=_decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    @staticmethod
    def _has_action_keys(result):
        """Check whether the response looks like a valid action chunk."""
        if not isinstance(result, dict) or not result:
            return False
        return any(k.startswith("action.") or k in ("action",) for k in result)

    def ping(self):
        """Check server connectivity."""
        try:
            request = {"endpoint": "ping"}
            self.sock.send(msgpack.packb(request, default=_encode))
            msgpack.unpackb(self.sock.recv(), object_hook=_decode)
            return True
        except zmq.error.ZMQError:
            return False

    def __del__(self):
        if hasattr(self, "sock"):
            self.sock.close()
        if hasattr(self, "ctx"):
            self.ctx.term()
