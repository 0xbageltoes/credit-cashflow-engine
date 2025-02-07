# Credit Cashflow Engine

A high-performance Python-based microservice for cash flow forecasting, built with FastAPI and integrated with Supabase.

## Features

- Cash flow forecasting with multiple scenarios
- Supabase JWT authentication
- Rate limiting and logging
- High-performance calculations using NumPy/Pandas
- Redis caching for frequent computations
- Docker deployment ready

## Setup

1. Create a `.env` file with your Supabase credentials:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_JWT_SECRET=your_jwt_secret
REDIS_URL=your_redis_url  # Optional
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- POST `/cashflow/forecast` - Run a cash flow forecast
- POST `/cashflow/scenario/save` - Save a forecasting scenario
- GET `/cashflow/scenario/load` - Load saved scenarios
- GET `/cashflow/history` - View forecast history

## Development

The project structure follows domain-driven design principles:
```
app/
├── api/           # API routes and endpoints
├── core/          # Core business logic and config
├── models/        # Pydantic models and database schemas
├── services/      # Business logic services
└── utils/         # Utility functions
```

## Deployment

1. Build Docker image:
```bash
docker build -t credit-cashflow-engine .
```

2. Run container:
```bash
docker run -p 8000:8000 credit-cashflow-engine
```
