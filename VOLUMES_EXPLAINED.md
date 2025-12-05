# 📁 Explicación de Volúmenes Docker

## 🎯 Respuesta Rápida

**Para el proyecto FUSION necesitas:**

```yaml
volumes:
  - /home/n8n:/home/n8n                      # ← REQUERIDO
  - paddlex-models:/home/n8n/.paddlex        # ← REQUERIDO
  - paddleocr-models:/home/n8n/.paddleocr    # ← REQUERIDO
  # OPCIONAL:
  # - /home/n8n/paddleocr-final/data/:/app/data/
```

## 📊 Comparación de Versiones

### Tu Proyecto Original (v2)

```yaml
volumes:
  - /home/n8n/paddleocr-final/data/:/app/data/
  - paddleocr-cpu-models:/app/.paddleocr
```

**Propósito:**
- `/app/data` → Datos de usuario/procesamiento
- `/app/.paddleocr` → Modelos PaddleOCR v2

### Proyecto de Paco (v3)

```yaml
volumes:
  - /home/n8n:/home/n8n
  - paddlex-models:/home/n8n/.paddlex
  - paddleocr-models:/home/n8n/.paddleocr
```

**Propósito:**
- `/home/n8n` → **TODO el workspace de n8n**
  - `/home/n8n/in/` → Archivos entrantes
  - `/home/n8n/ocr/` → Procesamiento intermedio
  - `/home/n8n/pdf/` → PDFs finales con OCR
  - `/home/n8n/json/` → Metadatos
- `/home/n8n/.paddlex` → Modelos PaddleX
- `/home/n8n/.paddleocr` → Modelos PaddleOCR v3

### Proyecto FUSION (nuevo)

```yaml
volumes:
  # REQUERIDO: Integración n8n
  - /home/n8n:/home/n8n

  # REQUERIDO: Cache de modelos
  - paddlex-models:/home/n8n/.paddlex
  - paddleocr-models:/home/n8n/.paddleocr

  # OPCIONAL: Datos adicionales
  # - /home/n8n/paddleocr-final/data/:/app/data/
```

## 🔍 Análisis Detallado

### Volumen 1: `/home/n8n:/home/n8n` (REQUERIDO)

**¿Por qué es necesario?**
- El endpoint `/ocr` de Paco espera archivos en `/home/n8n/in/`
- Los nuevos endpoints `/process` y `/analyze` guardan archivos temporales en `/home/n8n/in/`
- Los resultados se guardan en `/home/n8n/pdf/`
- Integración con workflows n8n

**¿Qué contiene?**
```
/home/n8n/
├── in/         ← Archivos entrantes (desde n8n o API)
├── ocr/        ← Procesamiento intermedio
├── pdf/        ← PDFs finales con capa OCR
├── json/       ← Metadatos (opcional)
├── .paddlex/   ← Modelos PaddleX (via volume)
└── .paddleocr/ ← Modelos PaddleOCR (via volume)
```

**Permisos:**
- En tu sistema: `/home/n8n` debe existir y tener permisos
- En el container: Se monta en `/home/n8n`

### Volumen 2: `paddlex-models:/home/n8n/.paddlex` (REQUERIDO)

**¿Por qué es necesario?**
- PaddleX descarga modelos grandes (~500MB+)
- Sin volume, se descargan cada vez que se reinicia el container
- Con volume, se persisten entre reinicios

**Tipo:** Named volume (gestionado por Docker)

**Primera vez:**
- El volume está vacío
- PaddleX descarga modelos automáticamente
- Tarda ~5-10 minutos (depende de conexión)

**Reinicios posteriores:**
- Los modelos ya están en el volume
- Container arranca en ~30 segundos

### Volumen 3: `paddleocr-models:/home/n8n/.paddleocr` (REQUERIDO)

**¿Por qué es necesario?**
- PaddleOCR descarga modelos de detección y reconocimiento
- Similar a PaddleX, evita redescargas

**Tipo:** Named volume (gestionado por Docker)

**Tamaño aproximado:** ~200-300MB

### Volumen 4: `/home/n8n/paddleocr-final/data/:/app/data/` (OPCIONAL)

**¿Lo necesitas?**
- ❓ Depende de si tu código usa `/app/data`
- ⚠️ El código de Paco NO usa `/app/data`
- ⚠️ Los nuevos endpoints NO usan `/app/data`

**Cuándo usarlo:**
- Si tienes scripts personales que escriben en `/app/data`
- Si quieres mantener compatibilidad con tu versión anterior
- Si guardas logs u otros archivos en `/app/data`

**Si no lo usas:**
- ✅ Todo funciona igual
- ✅ n8n workflows funcionan
- ✅ API REST funciona
- ✅ Solo no tendrás acceso a `/app/data` desde el host

## 🛠️ Configuración Recomendada

### Opción 1: Solo lo necesario (RECOMENDADO)

```yaml
volumes:
  - /home/n8n:/home/n8n
  - paddlex-models:/home/n8n/.paddlex
  - paddleocr-models:/home/n8n/.paddleocr
```

**Ventajas:**
- ✅ Mínimo y funcional
- ✅ Todo funciona (n8n + API REST)
- ✅ Menos complejidad

### Opción 2: Con datos adicionales

```yaml
volumes:
  - /home/n8n:/home/n8n
  - paddlex-models:/home/n8n/.paddlex
  - paddleocr-models:/home/n8n/.paddleocr
  - /home/n8n/paddleocr-final/data/:/app/data/
```

**Ventajas:**
- ✅ Compatible con scripts que usen `/app/data`
- ✅ Mantiene estructura de tu versión anterior

**Desventajas:**
- ⚠️ Un volume adicional innecesario si no se usa

### Opción 3: Datos dentro de /home/n8n (ALTERNATIVA)

```yaml
volumes:
  - /home/n8n:/home/n8n
  - paddlex-models:/home/n8n/.paddlex
  - paddleocr-models:/home/n8n/.paddleocr

# Y en lugar de /app/data, usa /home/n8n/data
# Ya está incluido en el primer volume
```

**Ventajas:**
- ✅ Todo centralizado en `/home/n8n`
- ✅ Solo 3 volumes
- ✅ Más simple de gestionar

## 📝 Verificar Estructura

### 1. Antes de arrancar Docker

Crear directorios en el host:

```bash
# Crear estructura n8n
sudo mkdir -p /home/n8n/in
sudo mkdir -p /home/n8n/ocr
sudo mkdir -p /home/n8n/pdf
sudo mkdir -p /home/n8n/json

# Dar permisos (opcional, depende de tu setup)
sudo chown -R $USER:$USER /home/n8n
sudo chmod -R 755 /home/n8n
```

### 2. Arrancar Docker

```bash
cd paddleocr_webcomunicav3_fusion
docker-compose up -d
```

### 3. Verificar volumes

```bash
# Ver volumes creados
docker volume ls | grep paddle

# Debería mostrar:
# paddleocr_webcomunicav3_fusion_paddlex-models
# paddleocr_webcomunicav3_fusion_paddleocr-models

# Ver contenido de /home/n8n dentro del container
docker exec paddlepaddle-cpu ls -la /home/n8n

# Debería mostrar:
# drwxr-xr-x  in/
# drwxr-xr-x  ocr/
# drwxr-xr-x  pdf/
# drwxr-xr-x  .paddlex/
# drwxr-xr-x  .paddleocr/
```

## ❓ FAQ

### ¿Puedo cambiar la ruta `/home/n8n` en el host?

**Sí**, pero debes cambiarla en:
1. `docker-compose.yml` → `volumes:` sección
2. Asegurar que el código espera `/home/n8n` DENTRO del container (no cambiar)

Ejemplo:
```yaml
volumes:
  - /mi/ruta/personalizada:/home/n8n  # ← Cambia la ruta del host
  # Dentro del container sigue siendo /home/n8n
```

### ¿Qué pasa si borro los named volumes?

```bash
docker volume rm paddleocr_webcomunicav3_fusion_paddlex-models
docker volume rm paddleocr_webcomunicav3_fusion_paddleocr-models
```

**Consecuencia:**
- Los modelos se borran
- En el siguiente `docker-compose up`, se descargan de nuevo
- Tarda ~5-10 minutos la primera vez

### ¿Necesito crear los volumes manualmente?

**NO**. Docker Compose los crea automáticamente cuando ejecutas:
```bash
docker-compose up -d
```

### ¿Puedo usar la misma estructura que mi v2?

**Sí**, pero tendrías DOS estructuras:
```yaml
volumes:
  # Para n8n y Paco (REQUERIDO)
  - /home/n8n:/home/n8n

  # Para tu estructura anterior (OPCIONAL)
  - /home/n8n/paddleocr-final/data/:/app/data/

  # Modelos (REQUERIDO)
  - paddlex-models:/home/n8n/.paddlex
  - paddleocr-models:/home/n8n/.paddleocr
```

**Pero es mejor simplificar:**
- Usa solo `/home/n8n` para todo
- Migra tus datos de `/app/data` a `/home/n8n/data`
- Menos volumes = más simple

## ✅ Recomendación Final

**Usa esta configuración:**

```yaml
volumes:
  - /home/n8n:/home/n8n
  - paddlex-models:/home/n8n/.paddlex
  - paddleocr-models:/home/n8n/.paddleocr
```

**Por qué:**
- ✅ Es lo mínimo necesario
- ✅ Todo funciona (n8n + API REST)
- ✅ Simple de mantener
- ✅ Compatible con el código de Paco
- ✅ Compatible con los nuevos endpoints

**Si realmente necesitas `/app/data`:**
- Añade la línea comentada en `docker-compose.yml`
- Descomenta: `# - /home/n8n/paddleocr-final/data/:/app/data/`

---

**¿Tienes dudas?** Pregunta antes de hacer `docker-compose up` para evitar problemas de permisos o rutas.
