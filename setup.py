from setuptools import setup

setup(name="fightin_words",
      version="1.0.0",
      description="Implementation of Monroe et al., 2008.",
      url="https://github.com/Wigder/fightin_words",
      author="Pedro Wigderowitz/Jack Hessel",
      author_email="pedrohm706@hotmail.com",
      python_requires=">=3.6.0",
      license="MIT",
      packages=["fightin_words"],
      install_requires=[
          "scikit-learn", "numpy"
      ],
      zip_safe=False)
