from setuptools import setup, find_packages

setup(
    name='utmos',
    version='1.0.2',
    requires=[
        'torch',
        'audio',
        'numpy',
        'fairseq',
        'cached-path',
        'click',
        'pytorch-lightning',
        'gradio'
    ],
    packages=find_packages(),
    author='mrfakename',
    author_email='me@mrfake.name',
    description='UT-Sarulab MOS prediction system using SSL models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ttseval/utmos',
    license='MIT',
    entry_points={
        "console_scripts": [
            "utmos = utmos.cli:main",
        ],
    },
)
