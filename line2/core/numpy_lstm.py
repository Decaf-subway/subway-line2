import io
import zipfile

import h5py
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _relu(x):
    return np.maximum(x, 0.0)


class NumpyLSTMModel:
    """Inference-only NumPy runner for the saved line2 LSTM models."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.weights = self._load_weights(model_path)

    @staticmethod
    def _read_dataset(group, path):
        return np.asarray(group[path], dtype=np.float32)

    @classmethod
    def _load_weights(cls, model_path):
        with zipfile.ZipFile(model_path) as zf:
            weights_bytes = zf.read("model.weights.h5")

        with h5py.File(io.BytesIO(weights_bytes), "r") as h5:
            return {
                "lstm0_kernel": cls._read_dataset(h5, "layers/lstm/cell/vars/0"),
                "lstm0_recurrent": cls._read_dataset(h5, "layers/lstm/cell/vars/1"),
                "lstm0_bias": cls._read_dataset(h5, "layers/lstm/cell/vars/2"),
                "lstm1_kernel": cls._read_dataset(h5, "layers/lstm_1/cell/vars/0"),
                "lstm1_recurrent": cls._read_dataset(h5, "layers/lstm_1/cell/vars/1"),
                "lstm1_bias": cls._read_dataset(h5, "layers/lstm_1/cell/vars/2"),
                "station_embedding": cls._read_dataset(h5, "layers/embedding/vars/0"),
                "dense0_w": cls._read_dataset(h5, "layers/dense/vars/0"),
                "dense0_b": cls._read_dataset(h5, "layers/dense/vars/1"),
                "dense1_w": cls._read_dataset(h5, "layers/dense_1/vars/0"),
                "dense1_b": cls._read_dataset(h5, "layers/dense_1/vars/1"),
            }

    @staticmethod
    def _lstm(x, kernel, recurrent_kernel, bias, return_sequences):
        x = np.asarray(x, dtype=np.float32)
        batch, steps, _ = x.shape
        units = recurrent_kernel.shape[0]
        h = np.zeros((batch, units), dtype=np.float32)
        c = np.zeros((batch, units), dtype=np.float32)
        outputs = []

        for t in range(steps):
            z = x[:, t, :] @ kernel + h @ recurrent_kernel + bias
            i = _sigmoid(z[:, :units])
            f = _sigmoid(z[:, units : 2 * units])
            c_bar = np.tanh(z[:, 2 * units : 3 * units])
            o = _sigmoid(z[:, 3 * units :])
            c = f * c + i * c_bar
            h = o * np.tanh(c)
            outputs.append(h)

        if return_sequences:
            return np.stack(outputs, axis=1)
        return h

    def __call__(self, inputs, training=False):
        del training

        seq = np.asarray(inputs[0], dtype=np.float32)
        station_idx = np.asarray(inputs[1]).astype(int).reshape(-1)

        w = self.weights
        x = self._lstm(
            seq,
            w["lstm0_kernel"],
            w["lstm0_recurrent"],
            w["lstm0_bias"],
            return_sequences=True,
        )
        x = self._lstm(
            x,
            w["lstm1_kernel"],
            w["lstm1_recurrent"],
            w["lstm1_bias"],
            return_sequences=False,
        )

        station_emb = w["station_embedding"][station_idx]
        x = np.concatenate([x, station_emb], axis=1)
        x = _relu(x @ w["dense0_w"] + w["dense0_b"])
        return x @ w["dense1_w"] + w["dense1_b"]
