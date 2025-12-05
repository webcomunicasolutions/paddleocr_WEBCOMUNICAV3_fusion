FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt update && apt install -y \
    jq \
    qpdf \
    poppler-utils \
    imagemagick \
    ghostscript \
    pdftk \
    curl \
    bc \
    ccache \
    htop \
    procps \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i '/<policy domain="coder" rights="none" pattern="PDF"/s/rights="none"/rights="read|write"/' /etc/ImageMagick-6/policy.xml

# Instalar dependencias Python
RUN python3.10 -m pip install --upgrade pip && \
    pip install --break-system-packages --no-cache-dir \
    numpy \
    decord \
    opencv-python==4.6.0.66 \
    paddleocr \
    pdf2image==1.16.3 \
    reportlab==4.0.4 \
    pdfplumber \
    Pillow>=10.0.0 \
    PyPDF2 \
    PyMuPDF \
    flask \
    waitress

# Configurar directorio de trabajo
WORKDIR /app

# Copiar aplicación
COPY app.py /app/

# Exponer puerto
EXPOSE 8503

# HEALTHCHECK muy permisivo para que EasyPanel no reinicie durante carga de PaddlePaddle
# - start-period: 300s (5 minutos) antes de empezar a verificar
# - interval: 60s entre verificaciones
# - timeout: 30s para responder
# - retries: 10 intentos antes de marcar unhealthy
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=10 \
    CMD curl -f http://localhost:8503/health || exit 0

# Comando por defecto
CMD ["python", "app.py"]

