# Portable package for the Cyclone Knowledge Platform.
#
# Contract Phase 8: this image must cold-build on a Mac mini today and a Mac
# Studio tomorrow with no source edits, so nothing here names a host path or a
# developer's home directory. Host-specific values arrive as environment
# variables at run time.

FROM python:3.12-slim

# Build-time provenance. CI passes the repo commit. Without it the image
# reports an unknown bundle_commit rather than a fabricated one -- git is not
# installed here, so the stamp is the only evidence the service can find.
ARG BUNDLE_COMMIT=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY fixtures/ ./fixtures/

RUN python -m pip install --no-cache-dir --disable-pip-version-check .

RUN if [ -n "$BUNDLE_COMMIT" ]; then \
      printf '%s\n' "$BUNDLE_COMMIT" > fixtures/synthetic-bundle/.bundle-commit; \
    fi

# Bind to every interface so the container is reachable; the package default
# stays on loopback so a local run is not accidentally exposed.
ENV CKP_SERVER_HOST=0.0.0.0 \
    CKP_SERVER_PORT=8080

RUN useradd --create-home --uid 10001 ckp && chown -R ckp:ckp /app
USER ckp

EXPOSE 8080
CMD ["python", "-m", "ckp"]
