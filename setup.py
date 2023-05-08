from setuptools import setup, find_packages

setup(
    name="zsrl",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "scikit-learn-extra>=0.2.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "lang2vec>=1.1.4",
    ],
    description=(
        "Zero-Shot Cross-Lingual Transfer Learning with Multiple Source and Target Languages "
        "for Information Extraction: Language Selection and Adversarial Training"
    ),
    author="Nghia Trung Ngo",
    license="MIT",
)
