#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base camera class.
"""

from __future__ import annotations

import abc
import uuid
from typing import Union

import numpy as np

__all__ = [
    "BaseCamera"
]


# MARK: - BaseCamera

# noinspection PyAttributeOutsideInit
class BaseCamera(metaclass=abc.ABCMeta):
    """Base Camera class.

    Attributes:
        id  (int, str):
            Camera's unique ID.
        dataset (str):
            Dataset name. It is also the name of the directory inside
            `data_dir`. Default: `None`.
        name (str):
            Camera name. It is also the name of the camera's config files.
            Default: `None`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        dataset: str,
        name   : str,
        id_    : Union[int, str] = uuid.uuid4().int,
        *args, **kwargs
    ):
        super().__init__()
        assert dataset is not None
        assert name    is not None
        self.id      = id
        self.dataset = dataset
        self.name    = name

    # MARK: Run

    @abc.abstractmethod
    def run(self):
        """Main run loop."""
        pass

    @abc.abstractmethod
    def run_routine_start(self):
        """Perform operations when run routine starts."""
        pass

    @abc.abstractmethod
    def run_routine_end(self):
        """Perform operations when run routine ends."""
        pass
