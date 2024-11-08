#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2024 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass, field
from pytransit.orbits import epoch

@dataclass
class Ephemeris:
    zero_epoch: float
    period: float
    duration: float

    def epoch(self, time: float) -> float:
        return epoch(time, self.zero_epoch, self.period)

    def transit_center(self, time: float) -> float:
        return self.zero_epoch + self.epoch(time)*self.period

    def transit_limits(self, time: float) -> tuple[float, float]:
        tc = self.transit_center(time)
        return tc-0.5*self.duration, tc+0.5*self.duration
