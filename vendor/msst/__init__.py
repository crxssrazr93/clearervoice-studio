"""
MelBandRoformer model code vendored from ZFTurbo's Music-Source-Separation-Training
(https://github.com/ZFTurbo/Music-Source-Separation-Training, MIT, see LICENSE here).

Source: models/bs_roformer/{mel_band_roformer,attend}.py at commit
6fdad13686bb7e84d029ec66f9dbe76e65c93ebf. The only local change is the Attend import,
made relative so the package works outside the MSST tree.
"""
from .mel_band_roformer import MelBandRoformer

__all__ = ["MelBandRoformer"]
