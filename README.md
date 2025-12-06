# PaddleOCR WebComunica V3 - FUSION Edition (Enfoque Correcto)

[![Version](https://img.shields.io/badge/version-3.0.0--fusion-blue.svg)](https://github.com/your-repo)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.x-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)

> **IMPORTANTE: REQUISITOS DE CPU**
>
> Este proyecto requiere una CPU con soporte **AVX/AVX2**. NO funcionará en:
> - VPS con "Common KVM processor" (virtualizados básicos)
> - CPUs antiguas sin instrucciones AVX
> - Algunos proveedores de hosting compartido
>
> **Error típico si la CPU no es compatible:** `Illegal instruction (core dumped)`
>
> Ver sección [Requisitos de CPU](#requisitos-de-cpu-importante) para más detalles.

## 🎯 Enfoque del Proyecto

**Este proyecto toma como BASE el proyecto de Paco (PaddleOCR v3 + preprocesamiento completo) y le añade la capa API REST del proyecto original.**

### ¿Qué es este proyecto?

```
BASE: Proyecto de Paco (paddlepaddle_paco)
  ├── PaddleOCR 3.x (vía PaddleX)
  ├── Preprocesamiento OpenCV completo
  ├── Corrección de perspectiva
  ├── Corrección de orientación
  ├── Corrección de inclinación
  ├── Procesamiento multi-página
  └── Integración n8n

  +

CAPA: API REST del Proyecto Original (PaddleOCRV2_WEBCOMUNICA)
  ├── Dashboard web interactivo
  ├── Endpoint /health completo
  ├── Endpoint /stats con métricas
  ├── Endpoint /process (wrapper REST)
  ├── Endpoint /analyze (análisis detallado)
  └── Monitoreo y estadísticas

  =

FUSION: Proyecto de Paco con API REST
  ✅ TODO el preprocesamiento de Paco
  ✅ API REST profesional del original
  ✅ Dashboard web para monitoreo
  ✅ Compatible con n8n
  ✅ Endpoints múltiples para distintos casos de uso
```

## 📊 Arquitectura

### Esquema de Capas

```
┌─────────────────────────────────────────────┐
│         API REST Layer (Añadido)            │
│  ┌──────┬──────┬────────┬─────────┬────┐  │
│  │  /   │/stats│/process│/analyze │... │  │
│  └──────┴──────┴────────┴─────────┴────┘  │
│                  ↓                          │
│         Wrapper/Translation Layer           │
│                  ↓                          │
│  ┌──────────────────────────────────────┐  │
│  │   Core Processing (Base de Paco)     │  │
│  │  • PaddleOCR 3.x (PaddleX)          │  │
│  │  • Preprocesamiento OpenCV          │  │
│  │  • Corrección perspectiva           │  │
│  │  • Corrección orientación           │  │
│  │  • Corrección inclinación           │  │
│  │  • Procesamiento multi-página       │  │
│  │  • Integración n8n                  │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Lo que SÍ se modificó

✅ **Añadido (sin tocar la base de Paco):**
- Dashboard web (`GET /`)
- Endpoint de estadísticas (`GET /stats`)
- Endpoint REST estándar (`POST /process`)
- Endpoint de análisis detallado (`POST /analyze`)
- Sistema de estadísticas del servidor
- Monitoreo y métricas

### Lo que NO se modificó

❌ **Mantenido intacto del proyecto de Paco:**
- Lógica de procesamiento OCR
- Preprocesamiento OpenCV
- Corrección de perspectiva
- Corrección de orientación
- Corrección de inclinación
- Procesamiento multi-página
- Integración n8n
- Endpoint `/ocr` original

## 🚀 Instalación

### Prerequisitos

- Docker y Docker Compose instalados
- 4GB RAM mínimo (8GB recomendado)
- **CPU con soporte AVX/AVX2** (ver sección de requisitos de CPU)

### Requisitos de CPU (IMPORTANTE)

PaddlePaddle requiere instrucciones AVX/AVX2 para funcionar. Si tu CPU no las soporta, obtendrás el error:

```
Illegal instruction (core dumped)
```

**Para verificar si tu CPU es compatible:**

```bash
# En Linux/Docker
cat /proc/cpuinfo | grep "model name" | head -1
cat /proc/cpuinfo | grep -o 'avx[^ ]*' | head -1
```

**CPUs NO compatibles:**
- `Common KVM processor` (VPS virtualizados básicos)
- CPUs muy antiguas sin AVX

**CPUs compatibles:**
- Intel Core i3/i5/i7/i9 (2011+)
- AMD Ryzen
- Intel Xeon (modernos)
- VPS con "dedicated CPU" o CPU passthrough

### Instalación Local (Docker Compose)

```bash
# Clonar repositorio
git clone https://github.com/webcomunicasolutions/paddleocr_WEBCOMUNICAV3_fusion.git
cd paddleocr_WEBCOMUNICAV3_fusion

# Construir e iniciar
docker-compose build
docker-compose up -d

# Verificar estado
curl http://localhost:8503/health
```

### Instalación en EasyPanel

> **ADVERTENCIA:** EasyPanel con servidores virtualizados básicos (KVM genérico)
> NO es compatible con PaddlePaddle. Antes de instalar, verifica que tu servidor
> tenga una CPU con soporte AVX/AVX2. Ver [Requisitos de CPU](#requisitos-de-cpu-importante).

#### Paso 1: Crear servicio desde GitHub

1. En EasyPanel, crear nuevo servicio "App"
2. Seleccionar "GitHub" como fuente
3. Repositorio: `webcomunicasolutions/paddleocr_WEBCOMUNICAV3_fusion`
4. Branch: `main`
5. Puerto: `8503`

#### Paso 2: Configurar variables de entorno

```env
# Flask
FLASK_ENV=production
FLASK_PORT=8503
TZ=Europe/Madrid

# OpenCV
OPENCV_HSV_LOWER_H=0
OPENCV_HSV_LOWER_S=0
OPENCV_HSV_LOWER_V=140
OPENCV_HSV_UPPER_H=180
OPENCV_HSV_UPPER_S=60
OPENCV_HSV_UPPER_V=255
OPENCV_MIN_AREA_PERCENT=0.05
OPENCV_EPSILON_FACTOR=0.01
OPENCV_ERODE_ITERATIONS=1
OPENCV_DILATE_ITERATIONS=2
OPENCV_MIN_WIDTH=300
OPENCV_MIN_HEIGHT=400
OPENCV_EROSION_PERCENT=0.085
OPENCV_INNER_SCALE_FACTOR=1.06

# Rotacion
ROTATION_MIN_CONFIDENCE=0.7
ROTATION_MIN_SKEW_ANGLE=0.2

# OCR
OCR_VERSION=PP-OCRv3
OCR_LANG=es
OCR_USE_DOC_ORIENTATION=false
OCR_USE_DOC_UNWARPING=false
OCR_USE_TEXTLINE_ORIENTATION=false
OCR_TEXT_DET_THRESH=0.1
OCR_TEXT_DET_BOX_THRESH=0.4
OCR_TEXT_DET_UNCLIP_RATIO=1.5
OCR_TEXT_DET_LIMIT_SIDE_LEN=960
OCR_TEXT_DET_LIMIT_TYPE=min
OCR_TEXT_RECOGNITION_BATCH_SIZE=6

# Optimizacion CPU
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
FLAGS_allocator_strategy=auto_growth
FLAGS_fraction_of_gpu_memory_to_use=0
CUDA_VISIBLE_DEVICES=""
```

#### Paso 3: Configurar volúmenes (Storage)

Crear 3 volúmenes persistentes:

| Nombre | Mount Path |
|--------|------------|
| `paddleocr-models` | `/home/n8n/.paddleocr` |
| `paddlex-models` | `/home/n8n/.paddlex` |
| `n8n-data` | `/home/n8n` |

#### Paso 4: Configurar dominio

1. Ir a "Domains" en EasyPanel
2. Añadir dominio (ej: `paddleocr.tu-dominio.easypanel.host`)
3. Puerto: `8503`

#### Paso 5: Build y Deploy

1. Hacer clic en "Deploy"
2. Esperar a que termine el build (~5-10 minutos)
3. Verificar que el contenedor está corriendo

### Verificar instalación

```bash
# Health check
curl https://tu-dominio/health

# Dashboard web
# Abrir en navegador: https://tu-dominio/

# Probar OCR (desde el dashboard o curl)
curl -X POST https://tu-dominio/process -F "file=@documento.pdf"
```

### Solución de problemas

#### Error: Contenedor se reinicia en bucle

**Causa:** El healthcheck de EasyPanel mata el contenedor antes de que cargue.

**Solución:** El código ya implementa carga on-demand. Los modelos se cargan con la primera petición OCR, no al inicio.

#### Error: `Illegal instruction (core dumped)`

**Causa:** La CPU del servidor no soporta instrucciones AVX/AVX2.

**Diagnóstico:**
```bash
# Dentro del contenedor
python3 -c "import paddle"
# Si da "Illegal instruction" -> CPU no compatible
```

**Solución:** Usar un servidor con CPU dedicada que soporte AVX2:

| Proveedor | Tipo de servidor | Compatible |
|-----------|------------------|------------|
| Hetzner | Dedicated CPU (CPX, CCX) | Si |
| DigitalOcean | Dedicated CPU Droplets | Si |
| AWS | C5, C6i (compute-optimized) | Si |
| Vultr | Dedicated Cloud | Si |
| OVH | Bare Metal / Dedicated | Si |
| Linode | Dedicated CPU | Si |
| EasyPanel (KVM genérico) | Shared CPU | **NO** |
| Otros VPS baratos | Shared/KVM básico | **Probablemente NO** |

**Cómo verificar ANTES de contratar:**
```bash
# Preguntar al proveedor si la CPU expone instrucciones AVX2
# O pedir un trial y ejecutar:
cat /proc/cpuinfo | grep avx
# Si no devuelve nada -> NO compatible
```

#### Error: `float() argument must be a string or a real number, not 'NoneType'`

**Causa:** Falta variable de entorno.

**Solución:** Verificar que todas las variables de entorno están configuradas (ver Paso 2).

#### Primera petición OCR muy lenta (~2 minutos)

**Esto es normal.** La primera petición carga PaddlePaddle y los modelos. Las siguientes peticiones serán rápidas (~1-2 segundos).

## 📖 Endpoints API

### GET `/` - Dashboard Web
Dashboard interactivo con métricas en tiempo real

**Características:**
- Estado del servidor
- Estadísticas de uso
- Ejemplos de uso
- Documentación de endpoints

### GET `/health` - Health Check
Health check completo del sistema

**Response:**
```json
{
  "status": "healthy",
  "ocr_ready": true,
  "preprocessor_ready": true,
  "version": "3.0.0-fusion",
  "uptime": 3600
}
```

### GET `/stats` - Estadísticas
Estadísticas detalladas del servidor

**Response:**
```json
{
  "uptime": 3600,
  "total_requests": 150,
  "successful_requests": 145,
  "failed_requests": 5,
  "success_rate": 96.67,
  "avg_processing_time": 1.2
}
```

### POST `/process` - OCR Estándar (Wrapper REST)
Endpoint REST que llama internamente al procesamiento de Paco

**Parámetros:**
- `file` (required): Archivo a procesar
- `language` (optional): Idioma (default: "es")
- `detailed` (optional): Respuesta detallada (default: false)

**Response:**
```json
{
  "success": true,
  "text": "Texto extraído...",
  "total_blocks": 85,
  "avg_confidence": 0.968,
  "processing_time": 1.2,
  "preprocessing_applied": true
}
```

### POST `/analyze` - Análisis Ultra-Detallado
Análisis visual detallado con información de cada bloque

**Parámetros:**
- `file` (required): Archivo a procesar
- `language` (optional): Idioma (default: "es")

**Response:**
```json
{
  "success": true,
  "ultra_analysis": "Texto formateado con indicadores visuales",
  "blocks": [
    {
      "text": "Texto del bloque",
      "confidence": 0.98,
      "orientation": "horizontal",
      "coordinates": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

### POST `/ocr` - Endpoint Original de Paco
Endpoint original de Paco para integración n8n (sin modificar)

**Parámetros:**
- `filename` (required): Path completo al archivo en /home/n8n

**Response:**
```json
{
  "success": true,
  "pdf_file": "documento.pdf",
  "pdf_path": "/home/n8n/pdf/documento.pdf",
  "extracted_text": "Texto extraído...",
  "stats": {
    "total_blocks": 85,
    "avg_confidence": 0.968
  }
}
```

## 🎯 Casos de Uso

### Caso 1: Integración n8n (usar endpoint original de Paco)

```bash
# Usar endpoint /ocr (sin modificar)
curl -X POST http://localhost:8503/ocr \
  -F "filename=/home/n8n/in/documento.pdf"
```

**Por qué:** Mantiene compatibilidad total con workflows n8n existentes.

### Caso 2: API REST estándar (usar nuevos endpoints)

```bash
# Usar endpoint /process (nuevo)
curl -X POST http://localhost:8503/process \
  -F "file=@documento.pdf" \
  -F "language=es" \
  -F "detailed=true"
```

**Por qué:** API REST estándar compatible con cualquier cliente HTTP.

### Caso 3: Análisis detallado para debugging

```bash
# Usar endpoint /analyze (nuevo)
curl -X POST http://localhost:8503/analyze \
  -F "file=@documento.pdf" \
  -F "language=es" | jq -r '.ultra_analysis'
```

**Por qué:** Visualización detallada de bloques detectados.

### Caso 4: Monitoreo del servidor

```bash
# Dashboard web
firefox http://localhost:8503/

# Estadísticas JSON
curl http://localhost:8503/stats | jq
```

**Por qué:** Monitoreo en tiempo real del estado del servidor.

## 🔧 Configuración

### Variables de Entorno

Todas las variables del proyecto de Paco se mantienen:

```bash
# OpenCV Configuration
OPENCV_HSV_LOWER_V=140
OPENCV_INNER_SCALE_FACTOR=1.12
# ... (todas las demás)

# Rotation Configuration
ROTATION_MIN_CONFIDENCE=0.7
ROTATION_MIN_SKEW_ANGLE=0.2

# n8n Integration
N8N_HOME_DIR=/home/n8n
```

### Docker Compose

El puerto se mantiene en **8503** para compatibilidad con proyecto de Paco:

```yaml
ports:
  - "8503:8503"

volumes:
  - /home/n8n:/home/n8n  # Integración n8n
  - paddleocr-models:/home/n8n/.paddleocr
  - paddlex-models:/home/n8n/.paddlex
```

## 📊 Comparación con Proyectos Base

### vs Proyecto Original (PaddleOCRV2_WEBCOMUNICA)

| Característica | Original v2 | FUSION v3 | Ventaja |
|----------------|-------------|-----------|---------|
| PaddleOCR Version | 2.8.1 | 3.x | ✅ FUSION (más reciente) |
| Preprocesamiento | ❌ No | ✅ Sí (completo) | ✅ FUSION |
| Corrección perspectiva | ❌ No | ✅ Sí | ✅ FUSION |
| Multi-página | ❌ No | ✅ Sí | ✅ FUSION |
| API REST | ✅ Sí (5 endpoints) | ✅ Sí (5 endpoints) | ⚖️ Igual |
| Dashboard | ✅ Sí | ✅ Sí | ⚖️ Igual |
| Integración n8n | ❌ No | ✅ Sí | ✅ FUSION |
| Configuración GANADORA | ✅ Sí | ❓ v3 (diferente API) | ❓ Por determinar |

### vs Proyecto de Paco (paddlepaddle_paco)

| Característica | Paco v3 | FUSION v3 | Ventaja |
|----------------|---------|-----------|---------|
| PaddleOCR Version | 3.x | 3.x | ⚖️ Igual |
| Preprocesamiento | ✅ Sí | ✅ Sí (idéntico) | ⚖️ Igual |
| Corrección perspectiva | ✅ Sí | ✅ Sí (idéntico) | ⚖️ Igual |
| Multi-página | ✅ Sí | ✅ Sí (idéntico) | ⚖️ Igual |
| API REST | ❌ No (solo /ocr) | ✅ Sí (5 endpoints) | ✅ FUSION |
| Dashboard | ❌ No | ✅ Sí | ✅ FUSION |
| Integración n8n | ✅ Sí | ✅ Sí (idéntico) | ⚖️ Igual |
| Estadísticas | ❌ No | ✅ Sí | ✅ FUSION |

**Conclusión**: FUSION = Proyecto de Paco + Dashboard + API REST + Estadísticas

## 🛠️ Gestión del Servidor

### Comandos Docker

```bash
# Iniciar
docker-compose up -d

# Detener
docker-compose down

# Ver logs
docker-compose logs -f

# Reiniciar
docker-compose restart

# Reconstruir
docker-compose build --no-cache
docker-compose up -d
```

### Verificación de Salud

```bash
# Health check básico
curl http://localhost:8503/health

# Estadísticas completas
curl http://localhost:8503/stats | jq

# Dashboard web
firefox http://localhost:8503/
```

## 📚 Documentación Adicional

- **CLAUDE.md** - Guía para desarrollo con Claude Code
- **PROYECTO_PACO_DOCUMENTACION.md** - Documentación completa del proyecto base
- **PROYECTO_ORIGINAL_DOCUMENTACION.md** - Documentación del proyecto original
- **PADDLEOCR_V2_VS_V3_EQUIVALENCIAS.md** - Equivalencias de API entre versiones

## ❓ Preguntas Frecuentes

### ¿Se modificó la lógica de Paco?

❌ **NO**. La lógica de procesamiento de Paco se mantiene 100% intacta. Solo se añadió una capa API REST encima.

### ¿Qué endpoints usar?

**Para n8n**: Usa `/ocr` (endpoint original de Paco)
**Para API REST**: Usa `/process` o `/analyze` (nuevos endpoints)
**Para monitoreo**: Usa `/`, `/health`, `/stats` (nuevos endpoints)

### ¿Es compatible con workflows n8n existentes?

✅ **SÍ**. El endpoint `/ocr` se mantiene idéntico. Workflows existentes funcionarán sin cambios.

### ¿Qué puerto usar?

**Puerto 8503** (mismo que proyecto de Paco para compatibilidad)

### ¿Se puede usar sin n8n?

✅ **SÍ**. Los nuevos endpoints REST (`/process`, `/analyze`) funcionan sin necesidad de estructura n8n.

## 🚀 Próximos Pasos

1. ✅ Probar compatibilidad con workflows n8n existentes
2. ⏳ Comparar rendimiento con proyecto original v2
3. ⏳ Documentar diferencias de precisión v2 vs v3
4. ⏳ Crear ejemplos de cliente Python
5. ⏳ Crear guía de migración desde proyecto original

## 📝 Changelog

### Version 3.0.0-fusion (2025-01-13)
- ✨ Proyecto base: paddlepaddle_paco (Paco)
- ✨ Añadido: Dashboard web interactivo
- ✨ Añadido: Endpoint `/stats` con métricas
- ✨ Añadido: Endpoint `/process` (wrapper REST)
- ✨ Añadido: Endpoint `/analyze` (análisis detallado)
- ✨ Añadido: Sistema de estadísticas del servidor
- ✅ Mantenido: 100% lógica de procesamiento de Paco
- ✅ Mantenido: Endpoint `/ocr` original (compatibilidad n8n)
- ✅ Mantenido: Toda configuración OpenCV de Paco

## 📄 Licencia

MIT License

## 🙏 Agradecimientos

- **Paco** por el excelente proyecto base con preprocesamiento avanzado
- **WebComunica** por la API REST y configuración GANADORA del proyecto original
- **PaddlePaddle Team** por el framework OCR
- **Claude Code** por la asistencia en el desarrollo

---

**Made with ❤️ by WebComunica + Paco + Claude Code**
**Enfoque: API REST sobre proyecto de Paco**
