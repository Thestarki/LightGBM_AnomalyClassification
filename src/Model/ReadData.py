"""Modulo para ler os dados resultantes da inferência."""
import sys

import pandas as pd

from config.config import settings

sys.path.append('src')
sys.path.append('Config')


def read_data() -> list:
    """
    Read the reference and malfuncion data.

    Read the reference and malfuncion data and
    save it in a list.

    Returns:
        list: A list with all the data in our problem.
    """
    dados = []
    for idx in range(len(list(settings.path_data.keys()))):
        dados.append(
            pd.read_csv(
                list(
                    settings.path_data.values(),
                    )[idx],
                    ),
                    )
    return dados
