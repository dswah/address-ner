import re
import random

import numpy as np
import faker


UNIT_WEIGHTS = (10000000*(2 ** (-np.arange(26, dtype=float)))).astype(int).tolist()
UNIT_VALUES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


class AddressGenerator:
    _STRASSE_PAT = re.compile("straße")

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._faker = faker.Faker("de")

        self._random = random
        self._random.seed(self.random_state)
        faker.Faker.seed(random_state)

    def _rand_bool(self, p=0.5):
        return self._random.random() < p

    def sample(self):

        street = self._faker.street_name()
        number = self._faker.building_number()
        unit = self._random.sample(UNIT_VALUES, counts=UNIT_WEIGHTS, k=1)[0]
        postcode = self._faker.postcode()
        city = self._faker.city()

        if self._rand_bool():
            street = self._STRASSE_PAT.sub("str.", street)

        out = ""
        out += street + " "
        out += number

        # empirical ratio of addresses with unit numbers
        if self._rand_bool(0.2):
            # almost always lowercase
            if self._rand_bool(0.1):
                unit = unit.upper()

            # sperated from house-number
            if self._rand_bool():
                out += " "
            out += unit

        # put everything on the same line
        if self._rand_bool():
            out += "\n"
        else:
            out += " "

        if self._rand_bool(0.8): # almost always add postcode and city
            out += postcode + " "
            out += city

        if self._rand_bool():
            out = out.lower()

        return out.strip()
