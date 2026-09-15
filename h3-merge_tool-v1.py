#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safetensors Model Merge Tool (MiniMax H3 / ComfyUI)
---------------------------------------------------
Merge two .safetensors models with RAM control (tensor by tensor),
support for int8 quantized tensors (ComfyUI format: weight + weight_scale
+ comfy_quant), optional GPU (CUDA) with CPU fallback.

Output formats:
  - auto: preserves the original dtype of each tensor (requantizes int8)
  - int8: quantizes conv/linear weights to int8 with per-row scale
  - fp8 : converts to float8_e4m3fn
  - fp16: converts to float16

Usage:  python merge_tool.py
"""

import gc
import json
import os
import queue
import struct
import threading
import traceback
import argparse

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

import torch

try:
    from safetensors import safe_open
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DTYPES_ST = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
DTYPES_ST_REV = {v: k for k, v in DTYPES_ST.items()}
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
# dtype used in merge calculations
MATH_DTYPE = torch.float32
# byte limit for processing a tensor on GPU (avoids VRAM OOM)
GPU_MAX_TENSOR_BYTES = 512 * 1024 * 1024


# ----------------------------------------------------------------------------
# Safetensors header reading (without loading weights)
# ----------------------------------------------------------------------------
def read_header(path):
    """Reads the JSON header of a .safetensors file. Returns (header, header_len)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def tensor_infos(header):
    """Returns {name: info} without the __metadata__."""
    return {k: v for k, v in header.items() if k != "__metadata__"}


# ----------------------------------------------------------------------------
# Individual tensor reading (streaming, without loading the whole file)
# ----------------------------------------------------------------------------
class TensorReader:
    """Reads tensors individually from a .safetensors via mmap."""

    def __init__(self, path):
        self.path = path
        header, self.header_len = read_header(path)
        self.header = header
        self.infos = tensor_infos(header)
        self._file = None
        self._mmap = None

    def _ensure_open(self):
        if self._file is None:
            import mmap
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, name):
        """Reads a tensor as torch.Tensor (original dtype)."""
        self._ensure_open()
        info = self.infos[name]
        dt = DTYPES_ST[info["dtype"]]
        start, end = info["data_offsets"]
        nbytes = end - start
        buf = self._mmap[self.header_len + start: self.header_len + end]
        t = torch.frombuffer(bytearray(buf), dtype=dt).reshape(info["shape"])
        return t

    def read_raw(self, name):
        """Reads raw bytes of a tensor (to copy without converting)."""
        self._ensure_open()
        info = self.infos[name]
        start, end = info["data_offsets"]
        return self._mmap[self.header_len + start: self.header_len + end]

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None


# ----------------------------------------------------------------------------
# Dequantization / Quantization (ComfyUI int8 format)
# ----------------------------------------------------------------------------
def dequantize_int8(weight_i8, scale):
    """weight (I8, [out, in]) * scale (F32, [out, 1]) -> F32."""
    return weight_i8.to(MATH_DTYPE) * scale.to(MATH_DTYPE)


def quantize_int8(weight_f32):
    """Quantizes to int8 with per-row scale. Returns (i8, scale_f32)."""
    w = weight_f32.to(MATH_DTYPE)
    if w.dim() == 1:
        w = w.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    if squeeze:
        q = q.squeeze(1)
        scale = scale.squeeze(1).unsqueeze(1)  # scale [n,1] like the originals
    else:
        scale = scale.reshape(-1, 1)
    return q, scale.to(torch.float32)


def cast_float(t, dtype_key):
    """Converts a float tensor to the target dtype."""
    if dtype_key == "F8_E4M3":
        return t.to(torch.float8_e4m3fn)
    if dtype_key == "F8_E5M2":
        return t.to(torch.float8_e5m2)
    return t.to(DTYPES_ST[dtype_key])


def tensor_to_bytes(t):
    """Serializes a tensor to bytes in safetensors format (little-endian)."""
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


def _comfy_quant_bytes():
    """Generates the comfy_quant descriptor (JSON bytes) for int8_tensorwise."""
    desc = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    return json.dumps(desc, separators=(",", ":")).encode("utf-8")


# ----------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------
def merge_values(a, b, wa, wb, method):
    """Merges two float tensors. method: 'linear' or 'weighted_sum'."""
    if method == "weighted_sum":
        total = wa + wb
        if total == 0:
            return a
        wa, wb = wa / total, wb / total
    return a * wa + b * wb


class MergeJob:
    """Runs the merge tensor by tensor and writes the output in streaming."""

    def __init__(self, path_a, path_b, out_path, weight_a, weight_b,
                 method="linear", out_format="auto", use_gpu=True,
                 log_fn=None, progress_fn=None, cancel_flag=None):
        self.path_a = path_a
        self.path_b = path_b
        self.out_path = out_path
        self.wa = weight_a / 100.0
        self.wb = weight_b / 100.0
        self.method = method
        self.out_format = out_format
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.log = log_fn or (lambda msg: None)
        self.progress = progress_fn or (lambda cur, total, name: None)
        self.cancel_flag = cancel_flag or (lambda: False)

    # ------------------------------------------------------------------ utils
    def _to_device(self, t):
        nbytes = t.numel() * t.element_size()
        if self.use_gpu and nbytes <= GPU_MAX_TENSOR_BYTES:
            return t.to(self.device, non_blocking=True)
        return t

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- validation
    def validate(self):
        ra = TensorReader(self.path_a)
        rb = TensorReader(self.path_b)
        try:
            ka, kb = set(ra.infos), set(rb.infos)
            only_a = ka - kb
            only_b = kb - ka
            if only_a:
                self.log(f"Warning: {len(only_a)} tensors only exist in A (will be kept from A).")
            if only_b:
                self.log(f"Warning: {len(only_b)} tensors only exist in B (will be kept from B).")
            mismatch = []
            for k in ka & kb:
                if ra.infos[k]["shape"] != rb.infos[k]["shape"]:
                    # comfy_quant descriptors may vary (JSON with spaces);
                    # they are regenerated during merge, so they don't block.
                    if k.endswith(".comfy_quant"):
                        continue
                    mismatch.append(k)
            if mismatch:
                raise ValueError(
                    f"{len(mismatch)} tensors with different shapes, e.g.: {mismatch[:3]}. "
                    "The models are not compatible for merging.")
            return ra, rb
        except Exception:
            ra.close()
            rb.close()
            raise

    # ------------------------------------------------------------------ merge
    def run(self):
        ra, rb = self.validate()
        tmp_data_path = self.out_path + ".dat"
        try:
            return self._run_merge(ra, rb, tmp_data_path)
        finally:
            ra.close()
            rb.close()
            if os.path.exists(tmp_data_path):
                try:
                    os.remove(tmp_data_path)
                except OSError:
                    pass

    def _run_merge(self, ra, rb, tmp_data_path):
        # Tensor order: A first, then the ones exclusive to B
        names = list(ra.infos.keys()) + [k for k in rb.infos if k not in ra.infos]
        total = len(names)

        out_header = {}
        offset = 0
        meta_a = ra.header.get("__metadata__", {})
        meta_b = rb.header.get("__metadata__", {})

        merged_meta = dict(meta_a)
        merged_meta["merge_tool"] = "safetensors_merge_tool"
        merged_meta["merge_parents"] = json.dumps(
            [os.path.basename(self.path_a), os.path.basename(self.path_b)])
        merged_meta["merge_weights"] = json.dumps(
            {"a": self.wa, "b": self.wb, "method": self.method})
        merged_meta["merge_format"] = self.out_format

        self.log(f"Device: {self.device} | Tensors: {total} | Method: {self.method} "
                 f"| Weights: A={self.wa:.0%} B={self.wb:.0%} | Format: {self.out_format}")

        with open(tmp_data_path, "wb") as data_f:
            for idx, name in enumerate(names):
                if self.cancel_flag():
                    raise InterruptedError("Cancelled by the user.")
                self.progress(idx, total, name)

                in_a = name in ra.infos
                in_b = name in rb.infos
                info_a = ra.infos.get(name)
                info_b = rb.infos.get(name)

                # ---- scale tensor: written together with the weight in
                #      _merge_quantized; here we just skip it.
                if name.endswith("_scale"):
                    base_weight = name[:-len("_scale")]
                    if base_weight in ra.infos or base_weight in rb.infos:
                        continue

                # ---- comfy_quant descriptor tensor: regenerated, since the
                #      merge recomputes the int8 weights (int8_tensorwise format).
                if name.endswith(".comfy_quant"):
                    base = name[:-len(".comfy_quant")] + ".weight"
                    if self._weight_is_i8(base, ra, rb):
                        raw = _comfy_quant_bytes()
                        data_f.write(raw)
                        out_header[name] = {
                            "dtype": "U8", "shape": [len(raw)],
                            "data_offsets": [offset, offset + len(raw)],
                        }
                        offset += len(raw)
                    continue

                # ---- int8 quantized weight (with weight_scale)
                scale_key = name + "_scale"
                is_quantized = ((info_a and info_a["dtype"] == "I8") or
                                (info_b and info_b["dtype"] == "I8")) and \
                               (scale_key in ra.infos or scale_key in rb.infos)

                if is_quantized:
                    offset = self._merge_quantized(
                        ra, rb, name, scale_key, in_a, in_b,
                        data_f, out_header, offset)
                    self._cleanup()
                    continue

                # ---- normal tensor (float/int)
                offset = self._merge_plain(
                    ra, rb, name, in_a, in_b, info_a, info_b,
                    data_f, out_header, offset)
                if (info_a and info_a["dtype"] in FLOAT_DTYPES) or \
                   (info_b and info_b["dtype"] in FLOAT_DTYPES):
                    self._cleanup()

            self.progress(total, total, "writing final file...")

        # ---- assemble final file: [len][json header][data]
        final_header = dict(out_header)
        final_header["__metadata__"] = merged_meta
        header_bytes = json.dumps(final_header, separators=(",", ":")).encode("utf-8")
        # safetensors requires header padding to a multiple of 8 (optional, but safe)
        pad = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * pad

        with open(tmp_data_path, "rb") as src, open(self.out_path, "wb") as dst:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        self.log(f"Done! File saved to: {self.out_path}")
        return self.out_path

    def _weight_is_i8(self, name, ra, rb):
        """True if the weight `name` will be written as int8 in the output."""
        if self.out_format not in ("auto", "int8"):
            return False
        scale_key = name + "_scale"
        info_a = ra.infos.get(name)
        info_b = rb.infos.get(name)
        is_quantized = ((info_a and info_a["dtype"] == "I8") or
                        (info_b and info_b["dtype"] == "I8"))
        return is_quantized and (scale_key in ra.infos or scale_key in rb.infos)

    # ---------------------------------------------------- quantized weights
    def _merge_quantized(self, ra, rb, name, scale_key, in_a, in_b,
                         data_f, out_header, offset):
        """Merge of int8 weight (dequant -> merge -> requant or float).
        Returns the new offset."""
        # dequantize A
        if in_a and scale_key in ra.infos:
            wa = dequantize_int8(ra.read(name), ra.read(scale_key))
        elif in_a:
            wa = ra.read(name).to(MATH_DTYPE)
        else:
            wa = None
        # dequantize B
        if in_b and scale_key in rb.infos:
            wb = dequantize_int8(rb.read(name), rb.read(scale_key))
        elif in_b:
            wb = rb.read(name).to(MATH_DTYPE)
        else:
            wb = None
        if wa is None:
            wa = wb.clone()
        if wb is None:
            wb = wa.clone()

        wa = self._to_device(wa)
        wb = self._to_device(wb)
        merged = merge_values(wa, wb, self.wa, self.wb, self.method).cpu()
        del wa, wb

        if self.out_format in ("auto", "int8"):
            q, scale = quantize_int8(merged)
            raw = tensor_to_bytes(q)
            data_f.write(raw)
            out_header[name] = {"dtype": "I8", "shape": list(q.shape),
                                "data_offsets": [offset, offset + len(raw)]}
            # scale always F32
            sraw = tensor_to_bytes(scale)
            data_f.write(sraw)
            out_header[scale_key] = {
                "dtype": "F32", "shape": list(scale.shape),
                "data_offsets": [offset + len(raw), offset + len(raw) + len(sraw)]}
            new_offset = offset + len(raw) + len(sraw)
        else:
            key = "F16" if self.out_format == "fp16" else "F8_E4M3"
            t = cast_float(merged, key)
            raw = tensor_to_bytes(t)
            data_f.write(raw)
            out_header[name] = {"dtype": key, "shape": list(t.shape),
                                "data_offsets": [offset, offset + len(raw)]}
            new_offset = offset + len(raw)
        del merged
        self._cleanup()
        return new_offset

    # ------------------------------------------------------- common tensors
    def _merge_plain(self, ra, rb, name, in_a, in_b, info_a, info_b,
                     data_f, out_header, offset):
        """Merge of a non-quantized tensor. Returns the new offset."""
        if in_a and in_b:
            ta = ra.read(name)
            tb = rb.read(name)
            dtype_a = info_a["dtype"]
            dtype_b = info_b["dtype"]
            if dtype_a in FLOAT_DTYPES and dtype_b in FLOAT_DTYPES:
                ta = self._to_device(ta.to(MATH_DTYPE))
                tb = self._to_device(tb.to(MATH_DTYPE))
                merged = merge_values(ta, tb, self.wa, self.wb, self.method).cpu()
                del ta, tb
                if self.out_format == "auto":
                    out_key = dtype_a
                elif self.out_format == "fp16":
                    out_key = "F16"
                elif self.out_format == "fp8":
                    out_key = "F8_E4M3"
                else:  # int8: keep original float dtype
                    out_key = dtype_a
                t = cast_float(merged, out_key)
                del merged
            elif dtype_a in FLOAT_DTYPES or dtype_b in FLOAT_DTYPES:
                # one float, the other non-float: merge in float
                ta = self._to_device(ta.to(MATH_DTYPE))
                tb = self._to_device(tb.to(MATH_DTYPE))
                t = merge_values(ta, tb, self.wa, self.wb, self.method).cpu()
                t = cast_float(t, dtype_a if dtype_a in FLOAT_DTYPES else dtype_b)
                del ta, tb
            else:
                # int/bool: keep from A
                t = ta
        else:
            src = ra if in_a else rb
            t = src.read(name)
            if self.out_format in ("fp16", "fp8") and \
               (info_a or info_b)["dtype"] in FLOAT_DTYPES:
                key = "F16" if self.out_format == "fp16" else "F8_E4M3"
                t = cast_float(t.to(MATH_DTYPE), key)

        raw = tensor_to_bytes(t)
        data_f.write(raw)
        out_key = DTYPES_ST_REV.get(t.dtype, "F32")
        out_header[name] = {"dtype": out_key, "shape": list(t.shape),
                            "data_offsets": [offset, offset + len(raw)]}
        new_offset = offset + len(raw)
        del t
        return new_offset


# ----------------------------------------------------------------------------
# Tkinter GUI
# ----------------------------------------------------------------------------
if _HAS_TK:
    class MergeApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Safetensors Model Merge — MiniMax H3")
            self.geometry("780x680")
            self.minsize(700, 600)

            self.msg_queue = queue.Queue()
            self.cancel_event = threading.Event()
            self.worker = None

            self._build_ui()
            self.after(100, self._poll_queue)

        # ------------------------------------------------------------- UI
        def _build_ui(self):
            pad = {"padx": 10, "pady": 5}
            main = ttk.Frame(self)
            main.pack(fill="both", expand=True)

            # --- Model A
            frm_a = ttk.LabelFrame(main, text="Model A")
            frm_a.pack(fill="x", **pad)
            self.var_a = tk.StringVar()
            self.cmb_a = ttk.Combobox(frm_a, textvariable=self.var_a, state="readonly")
            self.cmb_a.pack(fill="x", padx=8, pady=6)
            ttk.Button(frm_a, text="Browse...", command=lambda: self._browse(self.var_a)
                       ).pack(anchor="e", padx=8, pady=(0, 6))

            # --- Model B
            frm_b = ttk.LabelFrame(main, text="Model B")
            frm_b.pack(fill="x", **pad)
            self.var_b = tk.StringVar()
            self.cmb_b = ttk.Combobox(frm_b, textvariable=self.var_b, state="readonly")
            self.cmb_b.pack(fill="x", padx=8, pady=6)
            ttk.Button(frm_b, text="Browse...", command=lambda: self._browse(self.var_b)
                       ).pack(anchor="e", padx=8, pady=(0, 6))

            # --- Weights
            frm_w = ttk.LabelFrame(main, text="Merge ratio (sums to 100%)")
            frm_w.pack(fill="x", **pad)
            self.var_wa = tk.DoubleVar(value=50.0)
            self.var_wb = tk.DoubleVar(value=50.0)
            self._sync_lock = False

            ttk.Label(frm_w, text="Model A:").grid(row=0, column=0, padx=8, pady=6)
            self.scl_a = ttk.Scale(frm_w, from_=0, to=100, variable=self.var_wa,
                                   command=lambda v: self._sync_weights("a"))
            self.scl_a.grid(row=0, column=1, sticky="ew", padx=4)
            self.lbl_a = ttk.Label(frm_w, text="50%", width=6)
            self.lbl_a.grid(row=0, column=2, padx=8)

            ttk.Label(frm_w, text="Model B:").grid(row=1, column=0, padx=8, pady=6)
            self.scl_b = ttk.Scale(frm_w, from_=0, to=100, variable=self.var_wb,
                                   command=lambda v: self._sync_weights("b"))
            self.scl_b.grid(row=1, column=1, sticky="ew", padx=4)
            self.lbl_b = ttk.Label(frm_w, text="50%", width=6)
            self.lbl_b.grid(row=1, column=2, padx=8)
            frm_w.columnconfigure(1, weight=1)

            # --- Options
            frm_o = ttk.LabelFrame(main, text="Options")
            frm_o.pack(fill="x", **pad)
            ttk.Label(frm_o, text="Method:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
            self.var_method = tk.StringVar(value="linear")
            ttk.Combobox(frm_o, textvariable=self.var_method, state="readonly", width=22,
                         values=["linear", "weighted_sum"]).grid(row=0, column=1, sticky="w")

            ttk.Label(frm_o, text="Output format:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
            self.var_fmt = tk.StringVar(value="auto")
            ttk.Combobox(frm_o, textvariable=self.var_fmt, state="readonly", width=22,
                         values=["auto", "int8", "fp8", "fp16"]).grid(row=1, column=1, sticky="w")

            self.var_gpu = tk.BooleanVar(value=torch.cuda.is_available())
            self.chk_gpu = ttk.Checkbutton(
                frm_o, text="Use GPU (CUDA)" + ("" if torch.cuda.is_available() else " — not available"),
                variable=self.var_gpu,
                state="normal" if torch.cuda.is_available() else "disabled")
            self.chk_gpu.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=4)

            # --- Output
            frm_out = ttk.LabelFrame(main, text="Output file")
            frm_out.pack(fill="x", **pad)
            self.var_out = tk.StringVar()
            ttk.Entry(frm_out, textvariable=self.var_out).pack(
                fill="x", side="left", expand=True, padx=8, pady=8)
            ttk.Button(frm_out, text="Save as...",
                       command=self._browse_out).pack(padx=8, pady=8)

            # --- Progress
            frm_p = ttk.LabelFrame(main, text="Progress")
            frm_p.pack(fill="x", **pad)
            self.progress = ttk.Progressbar(frm_p, mode="determinate")
            self.progress.pack(fill="x", padx=8, pady=6)
            self.lbl_prog = ttk.Label(frm_p, text="Ready.")
            self.lbl_prog.pack(anchor="w", padx=8, pady=(0, 6))

            # --- Buttons
            frm_btn = ttk.Frame(main)
            frm_btn.pack(fill="x", **pad)
            self.btn_start = ttk.Button(frm_btn, text="Start Merge", command=self._start)
            self.btn_start.pack(side="left", padx=4)
            self.btn_cancel = ttk.Button(frm_btn, text="Cancel",
                                         command=self._cancel, state="disabled")
            self.btn_cancel.pack(side="left", padx=4)

            # --- Log
            frm_log = ttk.LabelFrame(main, text="Log")
            frm_log.pack(fill="both", expand=True, **pad)
            self.txt_log = tk.Text(frm_log, height=10, state="disabled", wrap="none")
            sb = ttk.Scrollbar(frm_log, command=self.txt_log.yview)
            self.txt_log.configure(yscrollcommand=sb.set)
            self.txt_log.pack(fill="both", expand=True, side="left", padx=8, pady=8)
            sb.pack(fill="y", side="right", pady=8)

            self._populate_models()

        def _populate_models(self):
            """Lists the .safetensors from the script directory and the cwd."""
            dirs = {os.path.dirname(os.path.abspath(__file__)), os.getcwd()}
            files = []
            for d in dirs:
                if os.path.isdir(d):
                    files += [os.path.join(d, f) for f in os.listdir(d)
                              if f.lower().endswith(".safetensors")]
            files = sorted(set(files))
            self.cmb_a["values"] = files
            self.cmb_b["values"] = files
            if files:
                self.cmb_a.current(0)
                self.cmb_b.current(min(1, len(files) - 1))

        # ------------------------------------------------------------ helpers
        def _browse(self, var):
            p = filedialog.askopenfilename(
                title="Select model",
                filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")])
            if p:
                var.set(p)

        def _browse_out(self):
            p = filedialog.asksaveasfilename(
                title="Save merged model", defaultextension=".safetensors",
                filetypes=[("Safetensors", "*.safetensors")])
            if p:
                self.var_out.set(p)

        def _sync_weights(self, changed):
            if self._sync_lock:
                return
            self._sync_lock = True
            try:
                if changed == "a":
                    va = self.var_wa.get()
                    self.var_wb.set(round(100.0 - va, 1))
                else:
                    vb = self.var_wb.get()
                    self.var_wa.set(round(100.0 - vb, 1))
                self.lbl_a.config(text=f"{self.var_wa.get():.0f}%")
                self.lbl_b.config(text=f"{self.var_wb.get():.0f}%")
            finally:
                self._sync_lock = False

        def _log(self, msg):
            self.msg_queue.put(("log", msg))

        def _poll_queue(self):
            try:
                while True:
                    kind, payload = self.msg_queue.get_nowait()
                    if kind == "log":
                        self.txt_log.configure(state="normal")
                        self.txt_log.insert("end", payload + "\n")
                        self.txt_log.see("end")
                        self.txt_log.configure(state="disabled")
                    elif kind == "progress":
                        cur, total, name = payload
                        self.progress.configure(maximum=total, value=cur)
                        self.lbl_prog.config(text=f"[{cur}/{total}] {name}")
                    elif kind == "done":
                        self.progress.configure(value=self.progress["maximum"])
                        self.lbl_prog.config(text="Done!")
                        self.btn_start.config(state="normal")
                        self.btn_cancel.config(state="disabled")
                        messagebox.showinfo("Merge", "Merge completed successfully!")
                    elif kind == "error":
                        self.lbl_prog.config(text="Error.")
                        self.btn_start.config(state="normal")
                        self.btn_cancel.config(state="disabled")
                        messagebox.showerror("Merge error", payload)
            except queue.Empty:
                pass
            self.after(100, self._poll_queue)

        # ------------------------------------------------------------ actions
        def _start(self):
            path_a = self.var_a.get()
            path_b = self.var_b.get()
            out_path = self.var_out.get()

            if not path_a or not os.path.isfile(path_a):
                messagebox.showerror("Error", "Select Model A.")
                return
            if not path_b or not os.path.isfile(path_b):
                messagebox.showerror("Error", "Select Model B.")
                return
            if path_a == path_b:
                messagebox.showerror("Error", "Models A and B must be different.")
                return
            if not out_path:
                messagebox.showerror("Error", "Set the output file.")
                return
            if os.path.exists(out_path) and not messagebox.askyesno(
                    "Overwrite", "The output file already exists. Overwrite?"):
                return

            self.cancel_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_cancel.config(state="normal")
            self.progress.configure(value=0)
            self._log(f"Starting merge: A={os.path.basename(path_a)} "
                      f"({self.var_wa.get():.0f}%) + B={os.path.basename(path_b)} "
                      f"({self.var_wb.get():.0f}%)")

            job = MergeJob(
                path_a, path_b, out_path,
                weight_a=self.var_wa.get(), weight_b=self.var_wb.get(),
                method=self.var_method.get(), out_format=self.var_fmt.get(),
                use_gpu=self.var_gpu.get(),
                log_fn=self._log,
                progress_fn=lambda c, t, n: self.msg_queue.put(("progress", (c, t, n))),
                cancel_flag=self.cancel_event.is_set,
            )
            self.worker = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            self.worker.start()

        def _run_job(self, job):
            try:
                job.run()
                self.msg_queue.put(("done", None))
            except InterruptedError:
                self.msg_queue.put(("log", "Merge cancelled."))
                self.msg_queue.put(("error", "Merge cancelled by the user."))
            except Exception as e:
                self.msg_queue.put(("log", traceback.format_exc()))
                self.msg_queue.put(("error", str(e)))

        def _cancel(self):
            self.cancel_event.set()
            self._log("Cancelling... (may take a few seconds)")


def main():
    parser = argparse.ArgumentParser(description="Merge safetensors models")
    parser.add_argument("--gui", action="store_true", help="Force graphical interface")
    parser.add_argument("--a", help="Model A (.safetensors)")
    parser.add_argument("--b", help="Model B (.safetensors)")
    parser.add_argument("--out", help="Output file")
    parser.add_argument("--wa", type=float, default=50.0, help="Weight of model A (%%)")
    parser.add_argument("--wb", type=float, default=50.0, help="Weight of model B (%%)")
    parser.add_argument("--method", default="linear", choices=["linear", "weighted_sum"])
    parser.add_argument("--format", default="auto", choices=["auto", "int8", "fp8", "fp16"])
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    if args.gui or (not args.a and not args.b):
        if not _HAS_TK:
            print("tkinter is not available in this Python. Use CLI mode:")
            print("  python merge_tool.py --a A.safetensors --b B.safetensors "
                  "--out out.safetensors --wa 60 --wb 40")
            return 1
        app = MergeApp()
        app.mainloop()
        return 0

    if not (args.a and args.b and args.out):
        parser.error("--a, --b and --out are required in CLI mode.")
    job = MergeJob(
        args.a, args.b, args.out,
        weight_a=args.wa, weight_b=args.wb,
        method=args.method, out_format=args.format,
        use_gpu=not args.cpu,
        log_fn=print,
        progress_fn=lambda c, t, n: print(f"\r[{c}/{t}] {n[:60]}", end=""),
    )
    job.run()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
