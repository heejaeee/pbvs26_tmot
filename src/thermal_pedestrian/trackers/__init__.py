# ==================================================================== #
# File name: __init__.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 19/07/2023
#
# ``tracker`` API consists of several trackers that share the same interface.
# Hence, they can be swap easily.
# ==================================================================== #

from __future__ import annotations

from .basetracker import BaseTracker
from .sort_adapter import *
from .boosttrack_adapter import *
from .diffmot_adapter import *
from .bytetrack_adapter import *
from .botsort_adapter import *
from .sam3_adapter import *
from .ocsort_adapter import *
