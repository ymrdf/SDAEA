"""Unix FD and CUDA-array transport for Godot external VkImages."""

from __future__ import annotations

import array
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile

import torch


_SOURCE = Path(__file__).with_name("gpu_vision") / "cuda_external_copy.cpp"


def _load_cuda_helper():
    """Compile the small Driver API wrapper once into the system temp directory."""
    import triton

    source = _SOURCE.read_bytes()
    key = hashlib.sha256(source + triton.__file__.encode()).hexdigest()[:16]
    output = Path(tempfile.gettempdir()) / f"sdaea_external_cuda_{key}.so"
    if not output.exists():
        include = Path(triton.__file__).parent / "third_party" / "cuda" / "include"
        temporary = output.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-fPIC", "-shared", str(_SOURCE),
             f"-I{include}", "-lcuda", "-o", str(temporary)],
            check=True,
        )
        os.replace(temporary, output)
    library = ctypes.CDLL(str(output))
    library.external_image_create.argtypes = (
        ctypes.c_int, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_uint,
    )
    library.external_image_create.restype = ctypes.c_void_p
    library.external_image_copy.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    library.external_image_copy.restype = ctypes.c_int
    library.external_image_destroy.argtypes = (ctypes.c_void_p,)
    return library


class ExternalCudaImage:
    def __init__(self, library, fd: int, size: int, width: int, height: int):
        self.library = library
        self.width = width
        self.height = height
        # Creating a tensor first initializes PyTorch's active CUDA context.
        torch.empty((), dtype=torch.uint8, device="cuda")
        self.handle = library.external_image_create(os.dup(fd), size, width, height)
        if not self.handle:
            raise RuntimeError("CUDA could not import a Godot OPAQUE_FD image")

    def copy_into(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cuda" or tensor.dtype != torch.uint8:
            raise ValueError("external image destination must be a CUDA uint8 tensor")
        if tuple(tensor.shape) != (self.height, self.width, 4) or not tensor.is_contiguous():
            raise ValueError("destination must be contiguous HxWx4 uint8")
        result = self.library.external_image_copy(self.handle, tensor.data_ptr())
        if result:
            raise RuntimeError(f"CUDA external image copy failed at stage {result}")

    def close(self) -> None:
        if self.handle:
            self.library.external_image_destroy(self.handle)
            self.handle = None


class GpuVisionTransport:
    def __init__(self, connection: socket.socket, metadata: dict, fds: list[int]):
        if metadata.get("magic") != "SDAEA_GPU_V1" or metadata.get("eyes") != 2:
            raise ValueError(f"Invalid GPU vision handshake: {metadata}")
        if metadata.get("vk_format") != 37 or len(fds) != 2:
            raise ValueError("GPU vision requires two RGBA8 OPAQUE_FD images")
        self.width = int(metadata["width"])
        self.height = int(metadata["height"])
        sizes = metadata["allocation_sizes"]
        if len(sizes) != 2 or self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid GPU vision dimensions or allocation sizes")
        self.agent_index = int(metadata.get("agent_index", 0))
        self.connection = connection
        self.library = _load_cuda_helper()
        self.images: list[ExternalCudaImage] = []
        try:
            for fd, size in zip(fds, sizes):
                self.images.append(ExternalCudaImage(
                    self.library, fd, int(size), self.width, self.height))
        except Exception:
            self.close()
            raise
        self.last_frame_id = None

    @classmethod
    def receive(cls, connection: socket.socket) -> "GpuVisionTransport":
        payload, ancillary, flags, _ = connection.recvmsg(4096, socket.CMSG_SPACE(2 * array.array("i").itemsize))
        descriptors = array.array("i")
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                usable = len(data) - (len(data) % descriptors.itemsize)
                descriptors.frombytes(data[:usable])
        try:
            if flags & socket.MSG_CTRUNC:
                raise RuntimeError("GPU vision descriptor message was truncated")
            chunks = [payload]
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(map(len, chunks)) > 65536:
                    raise ValueError("Oversized GPU handshake")
            metadata = json.loads(b"".join(chunks))
            return cls(connection, metadata, list(descriptors))
        finally:
            for fd in descriptors:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def copy_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = [torch.empty((self.height, self.width, 4),
                               dtype=torch.uint8, device="cuda") for _ in range(2)]
        for image, tensor in zip(self.images, tensors):
            image.copy_into(tensor)
        return tensors[0], tensors[1]

    def attach_observation(self, observation: dict, frame_id: int) -> None:
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            raise RuntimeError(f"GPU frame id did not advance: {self.last_frame_id} -> {frame_id}")
        left, right = self.copy_pair()
        observation["left_eye"] = left
        observation["right_eye"] = right
        self.last_frame_id = frame_id

    def close(self) -> None:
        for image in self.images:
            image.close()
        self.images.clear()

