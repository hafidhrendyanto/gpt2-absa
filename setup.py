from setuptools import setup, find_packages

setup(
    name='gpt2absa',
    version='0.1.0',
    author='Hafidh Rendyanto',
    author_email='hafidh.rendyanto@gmail.com',
    packages=find_packages(),
    license='MIT',
    description="Aspect Based Sentiment Analysis using OpenAI's GPT-2",
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "tensorflow",
        "transformers",
        "ipywidgets",
    ],
)