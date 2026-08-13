import os
import mat73 
import torch
import warnings
import scipy.io as sio
import numpy as np
import pandas as pd
import pynapple as nap
import compress_pickle
from dataclasses import fields, is_dataclass

warnings.filterwarnings("ignore", category=sio.matlab.MatWriteWarning)

def read_mat(file: str) -> dict[str, Any]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found, make sure you have the complete dataset.")
    try:
        return sio.loadmat(file, squeeze_me=True, struct_as_record=False)
    except:
        return mat73.loadmat(file)

TYPE_REGISTRY = {}

def register_type(cls):
    TYPE_REGISTRY[cls.__name__] = cls
    return cls

LOADERS = {}

def register_loader(name):
    def decorator(fn): 
        LOADERS[name] = fn
        return fn
    return decorator

WRITERS = {}

def register_writer(name):
    def decorator(fn): 
        WRITERS[name] = fn
        return fn
    return decorator



@register_loader("IntervalSet")
def intervalset_loader(obj) -> nap.IntervalSet:
    return nap.IntervalSet(
        obj["start"], 
        obj["end"]
    )

@register_writer("IntervalSet")
def intervalset_writer(obj) -> dict:
    return {
        "type": "IntervalSet",
        "start": obj.start,
        "end": obj.end
    }


@register_loader("Ts")
def ts_loader(obj) -> nap.Ts:
    return nap.Ts(
        t=obj["t"], 
        time_support=intervalset_loader(obj["time_support"])
    )

@register_writer("Ts")
def ts_writer(obj) -> dict:
    return {
        "type": "Ts",
        "t": obj.t,
        "time_support": intervalset_writer(obj.time_support)
    }


@register_loader("TsdFrame")
def tsd_loader(obj) -> nap.TsdFrame:
    return nap.TsdFrame(
        t=obj["t"],
        d=np.atleast_2d(obj["d"]),
        time_support=intervalset_loader(obj["time_support"]),
        columns=obj["columns"],
    )

@register_writer("TsdFrame")
def tsd_writer(obj) -> dict:
    return {
        "type": "TsdFrame",
        "t": obj.t,
        "d": obj.values,
        "time_support": intervalset_writer(obj.time_support),
        "columns": obj.columns,
    }


@register_loader("TsGroup")
def tsg_loader(obj) -> nap.TsGroup:
    return nap.TsGroup(
        {
            int(k): 
            ts_loader(v) for k,v in zip(obj['keys'], obj['values'])
        },
        time_support=intervalset_loader(obj['time_support'])
    )

@register_writer("TsGroup")
def tsg_writer(obj) -> dict:
    return {
        "type": "TsGroup",
        "keys": np.asarray(list(obj.keys()), dtype=np.int64),
        "values": [
            ts_writer(v) for v in obj.values()
        ],
        "time_support": intervalset_writer(obj.time_support)
    }


def save_pickle(data: Any, fname: str):
    s = compress_pickle.dumps(data, "gzip")
    with open(fname, 'wb') as f:
        f.write(s)

def read_pickle(fname: str):
    with open(fname, 'rb') as f:
        raw = f.read()
    return compress_pickle.loads(raw, "gzip")


def _to_mat_dict(obj: object):
    if obj is None:
        return np.array([])

    if isinstance(obj, (int, float, bool, str)):
        return obj

    if obj.__class__.__name__ in WRITERS:
        return WRITERS[obj.__class__.__name__](obj)

    if isinstance(obj, np.ndarray):
        return obj

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()

    if is_dataclass(obj):
        dclass = {
            field.name: _to_mat_dict(getattr(obj, field.name))
            for field in fields(obj)
        }
        dclass['type'] = obj.__class__.__name__
        return dclass

    if isinstance(obj, (list, tuple)):
        return np.array([_to_mat_dict(v) for v in obj], dtype=object)

    if isinstance(obj, dict):
        values = np.empty(len(obj), dtype=object)
        for i,v in enumerate(obj.values()):
            values[i] = _to_mat_dict(v)
        return {
            "type": "dict",
            "keys": np.asarray([
                k for k in obj.keys()
            ], dtype=object),
            "values": values,
        }

    if isinstance(obj, pd.DataFrame):
        return {
            "values": obj.to_numpy(),
            "index": np.asarray(obj.index),
            "columns": np.asarray(obj.columns, dtype=object),
        }

    if isinstance(obj, pd.Series):
        return {
            "values": obj.to_numpy(),
            "index": np.asarray(obj.index),
        }

    return str(obj)

def save_to_mat(file_path: str, dataclass: object):
    sio.savemat(
        file_path,
        _to_mat_dict(dataclass),
        do_compression=True,
        long_field_names=True
    )

def _mat_to_python(obj: object):
    """Recursively convert scipy mat_struct objects to dicts."""
    if hasattr(obj, "_fieldnames"):
        return {
            name: _mat_to_python(getattr(obj, name))
            for name in obj._fieldnames
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_mat_to_python(x) for x in obj.flat]

        return obj

    return obj

def _reconstruct_from_mat(obj):
    if not isinstance(obj, dict):
        if isinstance(obj, (list, tuple)):
            return [
                _reconstruct_from_mat(v) 
                for v in obj
            ]
        return obj

    type_name = obj.get("type")

    if type_name is None:
        return {
            k: _reconstruct_from_mat(v) 
            for k, v in obj.items()
        }

    if type_name == "dict":
        return {
            k: _reconstruct_from_mat(v) 
            for k, v in zip(obj["keys"], obj["values"])
        }

    if type_name in LOADERS:
        return LOADERS[type_name](obj)

    if type_name in TYPE_REGISTRY:

        cls = TYPE_REGISTRY[type_name]
        kwargs = {
            f.name: _reconstruct_from_mat(obj[f.name])
            for f in fields(cls)
            if f.name in obj
        }

        return cls(**kwargs)

    raise ValueError(f"Unknown type '{type_name}'")

def load_from_mat(file_path: str):
    mat = read_mat(
        file_path,
    )

    mat = {
        k: _mat_to_python(v)
        for k, v in mat.items()
        if not k.startswith("__")
    }

    return _reconstruct_from_mat(mat)

def _py2mat(obj):
    if isinstance(obj, dict):
        return {
            k: _py2mat(v)
            for k, v in obj.items()
        }
    else:
        data = compress_pickle.dumps(obj, "gzip")
        return np.frombuffer(data, dtype=np.uint8)

def save_to_mat2(file_path: str, dataclass: dict):
    """Save a dict to a file. Preserves all type metadata by 
    compressing everything as a python pickle file.
    """

    sio.savemat(
        file_path,
        _py2mat(dataclass),
        do_compression=True,
        long_field_names=True
    )

def _mat2py(obj):
    if isinstance(obj, dict):
        return {
            k: _mat2py(v)
            for k, v in obj.items()
        }
    else:
        data = obj.tobytes()
        return compress_pickle.loads(data, "gzip")

def load_from_mat2(file_path: str):
    """Load a dict from a file. Uncompresses the python pickle data.
    Works with files from `save_to_mat2`.
    """
    mat = read_mat(
        file_path,
    )
    mat = {
        k: _mat_to_python(v)
        for k, v in mat.items()
        if not k.startswith("__")
    }
    mat = {
        k: _mat2py(v)
        for k, v in mat.items()
        if not k.startswith("__")
    }
    return mat
