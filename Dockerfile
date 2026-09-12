FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS deps
COPY pyproject.toml README.md ./
COPY airfare ./airfare
RUN pip install --upgrade pip && pip install .

FROM base AS runtime
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY airfare ./airfare
COPY data/sample ./data/sample
COPY .streamlit ./.streamlit
RUN useradd --create-home app && mkdir -p data models && chown -R app:app /app
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"
CMD ["streamlit", "run", "airfare/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
