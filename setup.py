from setuptools import setup, find_packages

setup(
    name="content-moderation-openenv",
    version="1.0.0",
    description="AI Content Moderation Environment for OpenEnv",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "gunicorn>=21.0.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "gymnasium>=0.29.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.3.0",
        "openai>=1.0.0",
        "openenv-core>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "openenv-server=server.app:main",
        ],
    },
    python_requires=">=3.9",
)
