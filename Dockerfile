FROM python:3.12.13-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install --yes --no-install-recommends tzdata \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir \
	"ib-insync>=0.9.86" \
	"matplotlib>=3.9.0" \
	"pandas>=2.2.2" \
	"PyYAML>=6.0.1" \
	"python-dotenv>=1.0.1" \
	"requests>=2.32.3" \
	"tzdata>=2025.2"

COPY src ./src

RUN pip install --no-cache-dir --no-deps .

COPY config ./config

CMD ["trading-bot"]