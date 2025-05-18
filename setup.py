from setuptools import setup, find_packages

setup(
    name="analise_miopatia_ml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.2.2',
        'numpy>=1.26.4',
        'scikit-learn>=1.6.0',
        'tensorflow>=2.17.1',
        'keras>=3.5.0',
        'imbalanced-learn>=0.12.2',
        'fastapi>=0.111.0',
        'uvicorn>=0.30.1',
    ],
    python_requires='>=3.8',
)
