#!/usr/bin/env python3
"""GR00T inference client — thin ZMQ wrapper for policy server communication.

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

    def __init__(self, host="localhost", port=5555):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://{host}:{port}")

    def get_action(self, observations):
        """Send observations, receive action chunk."""
        request = {"endpoint": "get_action", "data": observations}
        self.sock.send(msgpack.packb(request, default=_encode))
        response = msgpack.unpackb(self.sock.recv(), object_hook=_decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

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
