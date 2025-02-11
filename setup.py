# setup.py at the root of your project (my_project/)
from setuptools import setup, find_packages

setup(
    name="llm_modules",
    version="0.1.0",
    author="Md Mushfiqur Rahman",
    author_email="mushfiqur11@iut-dhaka.edu",
    description="A very basic usage for LLMs with Huggingface and OpenAI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mushfiqur11/llm_module",
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        # List dependencies here, for example:
        "torch",
        "torchvision",
        "datasets",
        "openai",
        "peft",
        "pandas",
        "matplotlib",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)