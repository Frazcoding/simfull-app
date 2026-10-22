# simfull simulation software for Cloud Run (or any container host).
# The build context is this repository with the simcore checkout in vendor/simcore.
#
# Layer caching: dependencies (numpy, scipy, pydantic, etc.) are cached,
# then simcore's source, then the simfull-app source and web dashboard.
FROM mirror.gcr.io/library/python:3.12-slim

RUN pip install --no-cache-dir uv==0.11.32
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY vendor/simcore/pyproject.toml vendor/simcore/README.md /src/simcore/
COPY pyproject.toml uv.lock README.md /src/simfull-app/
WORKDIR /src/simfull-app
RUN uv sync --frozen --no-dev --no-install-project --no-install-package simcore && rm -rf /root/.cache

COPY vendor/simcore /src/simcore
COPY src /src/simfull-app/src
RUN uv sync --frozen --no-dev && rm -rf /root/.cache

ENV PATH=/src/simfull-app/.venv/bin:$PATH \
    SIMFULL_APP_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["simfull-app"]
