# Galaxy Zoo Classifier — dev container
# Python 3.10 (matching the original conda environment) with uv for dependency management.

FROM mcr.microsoft.com/devcontainers/python:3.10-bookworm

# Bring in uv as a single static binary from the official image.
# Pinned for reproducible builds — bump the tag to update uv.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

# UV_LINK_MODE=copy      — avoid hard-link warnings when installing onto the bind-mounted workspace
# UV_PYTHON_DOWNLOADS=never — use the image's Python 3.10 rather than a uv-managed download
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
