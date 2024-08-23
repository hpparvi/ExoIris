from pathlib import Path
from pytransit import LDTkLD as PTLDTkLD

class LDTkLD(PTLDTkLD):
    def __init__(self, pbs: tuple,
                 teff: tuple[float, float],
                 logg: tuple[float, float],
                 metal: tuple[float, float],
                 cache: str | Path | None = None,
                 dataset: str = 'vis-lowres'):
        super().__init__(pbs, teff, logg, metal, cache, dataset)
        self.dataset = dataset
