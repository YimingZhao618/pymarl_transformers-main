"""Frozen, reusable QMIX hypernetwork selector for the recovery protocol."""

import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F


class FrozenQMIXSelector:
    def __init__(self, checkpoint_path, n_agents, state_shape):
        path = os.path.join(checkpoint_path, "mixer.th") if os.path.isdir(checkpoint_path) else checkpoint_path
        if not os.path.isfile(path):
            raise FileNotFoundError("Frozen QMIX mixer checkpoint missing: {}".format(path))
        self.path = os.path.abspath(path)
        self.n_agents = int(n_agents)
        self.state_dim = int(np.prod(state_shape))
        with open(path, "rb") as stream:
            self.sha256 = hashlib.sha256(stream.read()).hexdigest()
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("QMIX selector checkpoint must contain a mixer state_dict")
        if "hyper_w_1.weight" in state:
            self.layers = [(state["hyper_w_1.weight"], state["hyper_w_1.bias"])]
        elif "hyper_w_1.0.weight" in state:
            self.layers = [
                (state["hyper_w_1.0.weight"], state["hyper_w_1.0.bias"]),
                (state["hyper_w_1.2.weight"], state["hyper_w_1.2.bias"]),
            ]
        elif "hyper_w1.0.weight" in state:
            self.layers = [
                (state["hyper_w1.0.weight"], state["hyper_w1.0.bias"]),
                (state["hyper_w1.2.weight"], state["hyper_w1.2.bias"]),
            ]
        else:
            raise ValueError("Checkpoint is not a supported QMIX first-layer hypernetwork")
        if self.layers[0][0].shape[1] != self.state_dim:
            raise ValueError("Selector state width {} != environment {}".format(
                self.layers[0][0].shape[1], self.state_dim))
        output_width = self.layers[-1][0].shape[0]
        if output_width % self.n_agents:
            raise ValueError("Selector output width {} is incompatible with {} agents".format(
                output_width, self.n_agents))
        self.embed_dim = output_width // self.n_agents
        self.layers = [(weight.detach().float(), bias.detach().float()) for weight, bias in self.layers]

    def score(self, initial_state):
        state = torch.as_tensor(np.asarray(initial_state, dtype=np.float32).reshape(-1))
        if state.numel() != self.state_dim:
            raise ValueError("Selector received the wrong global state shape")
        with torch.no_grad():
            output = state
            for index, (weight, bias) in enumerate(self.layers):
                output = F.linear(output, weight, bias)
                if index + 1 < len(self.layers):
                    output = F.relu(output)
            scores = output.reshape(self.n_agents, self.embed_dim).abs().mean(dim=-1).numpy()
        if not np.isfinite(scores).all():
            raise ValueError("QMIX selector produced non-finite scores")
        return scores

    def select(self, initial_state, alive_mask):
        scores = self.score(initial_state)
        alive = np.asarray(alive_mask, dtype=bool)
        if alive.shape != (self.n_agents,) or not alive.any():
            raise ValueError("Invalid initial alive mask for QMIX selector")
        scores[~alive] = -np.inf
        return int(np.argmax(scores)), scores
