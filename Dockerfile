# Portable package for the Cyclone Knowledge Platform.
#
# Contract Phase 8: this image must cold-build on a Mac mini today and a Mac
# Studio tomorrow with no source edits, so nothing here names a host path or a
# developer's home directory. Host-specific values arrive as environment
# variables at run time.

FROM python:3.12-slim AS runtime

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

# P1 pilot corpus binding (issue #23) is benchmark-support tooling over the
# real Wiki checkout, not part of the served API -- it never ships in the
# runtime image.
RUN rm -rf ./src/ckp/pilot

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

# Disposable Linux-only C7 verification stage. The shipped runtime above
# intentionally has neither Git nor dev dependencies; this stage adds them
# only so CI can exercise isolated synthetic Git transactions in a container.
FROM runtime AS c7-writer-smoke

USER root
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
RUN python -m pip install --no-cache-dir --disable-pip-version-check ".[dev,index]"
USER ckp

# Keep an ordinary `docker build` on the production runtime, not the C7 test
# stage. `scripts/smoke-container.sh` selects c7-writer-smoke explicitly.
FROM runtime AS final
