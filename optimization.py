import torch


__all__ = ["AbsVolumeWrapper"]


r"""
Provides wrapper classes for optimising detectors and other quality-of-life methods

FitParams and AbsVolumeWrapper are modified versions of the FitParams in LUMIN (https://github.com/GilesStrong/lumin/blob/v0.7.2/lumin/nn/models/abs_model.py#L16)
and Model in LUMIN (https://github.com/GilesStrong/lumin/blob/master/lumin/nn/models/model.py#L32), distributed under the following licence:
    Copyright 2018 onwards Giles Strong

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Usage is compatible with the AGPL licence under which OPPAC_opt is distributed.
Stated changes: adaption of FitParams to pass type-checking, heavy adaptation of Model to be suitable for task specific training
"""




# Lots of work to be done here
