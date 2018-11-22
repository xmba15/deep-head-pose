# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import torch


_DIRECTORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))
_MODEL_PATH = os.path.join(_DIRECTORY_ROOT, "models")
_MODEL = os.path.join(_MODEL_PATH, "hopenet_robust_alpha1.pkl")
_IMAGE_PATH = os.path.join(_DIRECTORY_ROOT, "images")
_TEST_IMAGE = os.path.join(_IMAGE_PATH, "test_img.jpg")


class Config(object):
    def __init__(self):
        self.MODEL = _MODEL
        self.USE_CUDA = torch.cuda.is_available()
        self.TEST_IMAGE = _TEST_IMAGE

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


def main():
    config = Config()
    config.display()


if __name__ == '__main__':
    main()
