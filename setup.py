from setuptools import setup, find_packages


setup(
    name='chat_in_a_nutshell',
    version='1.5.1',
    author='David Novák',
    description="A terminal-based script for interacting with OpenAI's and/or Anthropic's API.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thedavidnovak/chat-in-a-nutshell',
    packages=find_packages(),
    py_modules=['chat_provider'],
    entry_points={
        'console_scripts': [
            'ch=chat_provider:main',
        ],
    },
    python_requires='>=3.8',
    install_requires=[
        'openai~=1.62.0',
        'anthropic~=0.49.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)