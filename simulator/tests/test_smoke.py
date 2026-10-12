import numpy as np
import pandas as pd
import yaml


def test_environment_imports():
    assert np.__version__ and pd.__version__ and yaml.__version__
