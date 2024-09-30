"""Configuration File of the data."""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv = os.path.join(os.path.dirname(__file__), '.env')


class Settings(BaseSettings):
    """Sets all the data settings used in the model.

    Args:
        Base (_type_): Pydantinc BaseSettings.
    """

    model_config = SettingsConfigDict(env_file=dotenv, env_file_encoding='utf-8')
    variables: list

    frac_train: float
    frac_test: float
    frac_val: float

    seed: int
    params: dict
    num_round: int

    path_data: dict


settings = Settings()
