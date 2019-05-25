import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sdss_gz_data',  
    version='0.1',
    author="Will Gauvin",
    author_email="wgauvin@gmail.com",
    description="A utility package dealing with the Galaxy Zoo (GZ) from the Sloan Digital Sky Survey (SDSS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wgauvin/astroml",
    packages=setuptools.find_packages(),
    license='MIT',
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'keras',
        'google-cloud-storage'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )
