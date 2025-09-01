FROM ghcr.io/astral-sh/uv:python3.12-alpine

# Set working directory
WORKDIR /app

# Copy the rest of the application files
COPY . .

RUN apk add --no-cache git
# Install dependencies
RUN uv pip install --system -e .

# Expose app port
EXPOSE 8001

# Start the application
ENV HOST=0.0.0.0 PORT=8001
CMD ["python", "main.py"]