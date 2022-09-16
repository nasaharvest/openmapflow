import collections
import functools
import os
from datetime import date
from typing import Union

import numpy as np
import pandas as pd
from pandas.compat._optional import import_optional_dependency

try:
    import google.colab  # noqa F401
    from tqdm.notebook import tqdm  # noqa F401

    IN_COLAB = True
except ImportError:
    from tqdm import tqdm  # noqa F401

    IN_COLAB = False


def colab_gee_gcloud_login(project_id: str):
    google_package = import_optional_dependency("google")
    ee = import_optional_dependency("ee")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    print("Logging into Google Cloud")
    google_package.colab.auth.authenticate_user()
    print("Logging into Earth Engine")
    SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/earthengine",
    ]
    CREDENTIALS, _ = google_package.auth.default(default_scopes=SCOPES)
    ee.Initialize(CREDENTIALS, project=project_id)


def confirmation(message, default="y"):
    print(message)
    if default == "y":
        return input("Confirm [y]/n: ").lower() != "n"
    else:
        return input("Confirm y/[n]: ").lower() == "y"


def to_date(d):
    if type(d) == np.datetime64:
        return d.astype("M8[D]").astype("O")
    elif type(d) == str:
        return pd.to_datetime(d).date()
    else:
        return d.date()


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def str_to_np(x: str) -> np.ndarray:
    return np.array(eval(x))
