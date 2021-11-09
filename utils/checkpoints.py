"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import sys
import torch


class Checkpoint(object):

    def __init__(self, path, max_n=3):
        self.path = path
        self.max_n = max_n
        self.models = {}
        self.checkpoints = []

    def create_model(self, model, opt):
        self.models = {}
        self.models['model'] = model
        self.models['opt'] = opt

    def limit(self):
        return self.max_n

    def __len__(self):
        return len(self.checkpoints)

    def _get_full_path(self, suffix):
        filename = self._filename(suffix)
        return os.path.join(self.path, filename)

    def clean(self):
        n_remove = max(0, len(self.checkpoints) - self.max_n)
        for i in range(n_remove):
            self._rm(self.checkpoints[i])
        self.checkpoints = self.checkpoints[n_remove:]

    def _rm(self, suffix):
        path = self._get_full_path(suffix)
        if os.path.isfile(path):
            os.remove(path)

    def _filename(self, suffix):
        return "{}.pth".format(suffix)

    def load(self, path, location):
        if len(path) > 0 and not os.path.isfile(path):
            print("Snapshot {} not found".format(path))

        data = torch.load(path, map_location=location)
        data_mapped = {}
        for key, val in data["model"].items():
            data_mapped[key.replace("module.", "")] = val

        self.models["model"].load_state_dict(data_mapped, strict=True)
        return data["epoch"], data["score"]

    def checkpoint(self, score, epoch, t):
        suffix = "epoch{:03d}_score{:4.3f}_{}".format(epoch, score, t)
        self.checkpoints.append(suffix)

        path = self._get_full_path(suffix)
        if not os.path.isfile(path):
            torch.save({"model": self.models["model"].state_dict(),
                        "opt": self.models["opt"].state_dict(),
                        "score": score,
                        "epoch": epoch}, path)

        # removing if more than allowed number of snapshots
        self.clean()
