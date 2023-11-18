from setuptools import setup

setup(
    name="vocoders",
    version="0.0.1",
    packages=["vocoders"],
    install_requires=[
        "torch",
        "torchaudio",
        "lightning",
        "hydra-core",
        "matplotlib",
        "pandas",
        "tqdm",
        "pyworld",
        "joblib",
    ],
)
