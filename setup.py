from setuptools import find_packages, setup


VERSION = "0.0.1"
DESCRIPTION = "Reference implementation and notebooks for WeightedERM"
LONG_DESCRIPTION = (
    "Reference implementation and notebooks for WeightedERM (WERM), "
    "for high-dimensional change point detection."
)


setup(
    name="WERM",
    version=VERSION,
    author="Gabriel Arpino",
    author_email="ga442@cam.ac.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "jax",
        "jaxlib",
        "jaxopt",
        "scikit-learn",
        "tqdm",
        "joblib",
        "torch",
    ],
    extras_require={
        "notebooks": [
            "notebook",
            "pandas",
            "ucimlrepo",
            "seaborn",
            "KDEpy",
            "rpy2",
        ],
    },
    keywords=["python", "changepoint detection", "weighted erm"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.10.10",
)
