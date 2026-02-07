FROM python:3.13-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Copy dependency files first (optimizes Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies 
# Setting --frozen ensures uv.lock isn't updated during build
ENV UV_PROJECT_ENVIRONMENT=/code/venv
# Add the virtual environment to the PATH 
ENV PATH="/code/venv/bin:$PATH"
RUN uv sync --frozen --no-dev

# Copy the rest of the source code
COPY . .
# and install it
RUN uv pip install -e .

# Mount dataset path
VOLUME ["/data"]
# Expose Streamlit port. Please make sure that port 8501 is free.
EXPOSE 8501

# Run the streamlit application
CMD ["streamlit", "run", "src/app/Hello.py", "--server.port=8501", "--server.address=0.0.0.0"]
