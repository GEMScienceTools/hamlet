from typing import Optional

from openquake.hazardlib.geo.point import Point


class SimpleRupture():
    __slots__ = [
        'strike', 'dip', 'rake', 'mag', 'hypocenter', 'occurrence_rate',
        'source'
    ]

    def __init__(
            self,
            strike: Optional[float] = None,
            dip: Optional[float] = None,
            rake: Optional[float] = None,
            mag: Optional[float] = None,
            hypocenter: Optional[Point] = None,
            occurrence_rate: Optional[float] = None,
            source: Optional[str] = None,
    ):
        self.strike = strike
        self.dip = dip
        self.rake = rake
        self.mag = mag
        self.hypocenter = hypocenter
        self.occurrence_rate = occurrence_rate
        self.source = source
