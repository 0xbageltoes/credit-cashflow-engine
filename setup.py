from setuptools import setup, find_packages

setup(
    name="credit-cashflow-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-jose[cryptography]==3.3.0",
        "supabase==2.0.3",
        "numpy==1.26.2",
        "numpy-financial==1.0.0",
        "pandas==2.1.3",
        "python-dotenv==1.0.0",
        "redis==5.0.1",
        "pydantic==2.5.2",
        "pydantic-settings==2.1.0",
        "python-multipart==0.0.6",
        "httpx>=0.24.0,<0.25.0",
        "celery==5.3.6",
        "websockets==12.0",
        "aioredis==2.0.1",
        "flower==2.0.1",
        "prometheus-client==0.19.0",
        "scipy==1.11.4",
        "statsmodels==0.14.1"
    ],
    extras_require={
        "test": [
            "pytest==7.4.3",
            "pytest-asyncio==0.21.1",
            "pytest-cov==4.1.0",
            "pytest-mock==3.12.0"
        ]
    }
)
