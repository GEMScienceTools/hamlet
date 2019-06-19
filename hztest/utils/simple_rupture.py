class SimpleRupture():
    __slots__ = ['rake', 'mag', 'hypocenter', 'occurrence_rate', 'source']

    def __init__(self, rake, mag, hypocenter, occurrence_rate, source):
        self.rake = rake
        self.mag = mag
        self.hypocenter = hypocenter
        self.occurrence_rate = occurrence_rate
        self.source = source
