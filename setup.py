from setuptools import setup

setup(name='fightin_words',
      version='0.1',
      description='Implementation of Monroe et al., 2008.',
      url='https://github.com/Wigder/fightin_words',
      author='Pedro Wigderowitz/Jack Hessel',
      author_email='pedrohm706@hotmail.com',
      license='MIT',
      packages=['fightin_words'],
      install_requires=[
          'scikit-learn',
      ],
      zip_safe=False)
