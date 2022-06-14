"""setup.py file for packaging transferprediction."""

from setuptools import setup


with open('readme.md', 'r') as readme_file:
    readme = readme_file.read()


setup(
    name='transferprediction',
    version='0.0.1',
    description='examining transfer learning methods in natural language processing',
    long_description=readme,
    url='https://github.com/orionw/transferprediction',
    author='Orion Weller',
    keywords='machine-learning, ml, transfer-learning, natural-language-processing, ai',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache',
    packages=['transferprediction'],
    package_dir={'': 'src'},
    scripts=[],
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    python_requires='>= 3.6',
    zip_safe=False)
