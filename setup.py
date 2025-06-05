from setuptools import setup, find_packages

setup(
    name="ml_recommendation_engine",
    version="0.1.0",
    description="Film ve TV Dizisi Öneri Sistemi",
    author="MovieApp Team",
    packages=find_packages(),
    install_requires=[
        "flask==2.0.3",
        "werkzeug==2.0.3",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "requests==2.27.1",
        "pyjwt==2.4.0",
        "python-dotenv==0.20.0",
        "matplotlib==3.5.1",
        "seaborn==0.11.2",
        "joblib==1.1.0",
        "scipy==1.7.3"
    ],
    python_requires=">=3.7, <3.9",
) 