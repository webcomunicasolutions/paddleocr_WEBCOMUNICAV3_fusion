# CLAUDE.md - Development Guide

This file provides guidance to Claude Code when working with the PaddleOCR Fusion v3 codebase.

## Project Overview

**PaddleOCR Fusion v3** is an API REST layer built on top of Paco's PaddleOCR 3.x project. It maintains 100% of Paco's processing logic while adding professional REST endpoints for easier integration.

## Architecture

```
┌─────────────────────────────────────────────┐
│         API REST Layer (Added)              │
│  Lines 1774-2312                            │
│  ┌──────┬──────┬────────┬─────────┬────┐  │
│  │  /   │/stats│/process│/analyze │... │  │
│  └──────┴──────┴────────┴─────────┴────┘  │
│                  ↓                          │
├─────────────────────────────────────────────┤
│   Core Processing (Paco's Base - Untouched)│
│  Lines 1-1773                               │
│  • PaddleOCR 3.x via PaddleX               │
│  • OpenCV preprocessing                     │
│  • Perspective correction                   │
│  • Orientation correction                   │
│  • Deskew correction                        │
│  • Multi-page processing                    │
│  • n8n integration                          │
│  • Original /ocr endpoint                   │
└─────────────────────────────────────────────┘
```

## Key Files Structure

### app.py (2331 lines total)

**Lines 1-1773: Paco's Base (UNTOUCHED)**
- All imports and configuration
- OpenCV preprocessing functions
- PaddlePaddle orientation correction
- ImageMagick deskew
- PDF processing functions
- `/health` endpoint (line 1635)
- `/ocr` endpoint (line 1647-1772)

**Lines 1774-2312: API REST Layer (ADDED)**
- `server_stats` dictionary (line 1780)
- `GET /` - Dashboard (line 1788)
- `GET /stats` - Statistics (line 2022)
- `POST /process` - REST wrapper (line 2053)
- `POST /analyze` - Detailed analysis (line 2186)
- Startup logging enhancements (line 2314)

### Other Files

- `Dockerfile` - From Paco (unchanged)
- `docker-compose.yml` - From Paco (unchanged)
- `README.md` - New documentation
- `.env.example` - Environment variables template
- `.gitignore` - Git exclusions

## Important Notes

### What NOT to Modify

❌ **DO NOT modify lines 1-1773** unless fixing critical bugs. This is Paco's proven processing logic.

### What You Can Modify

✅ **Lines 1774-2312**: API REST layer
- Add new endpoints
- Modify dashboard HTML
- Add statistics
- Improve error handling

## Adding New Endpoints

Template for adding new REST endpoints (add after line 2307):

```python
@app.route('/your-endpoint', methods=['POST'])
def your_endpoint():
    """Your endpoint description"""
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1

    temp_file_path = None

    try:
        # 1. Validate file
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'No file provided'}), 400

        # 2. Save to /home/n8n/in
        n8nHomeDir = '/home/n8n'
        temp_filename = f"temp_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        # 3. Call /ocr internally
        with app.test_request_context('/ocr', method='POST',
                                     data={'filename': temp_file_path}):
            response = ocr()
            response_json = response.get_json() if not isinstance(response, tuple) else response[0].get_json()

        # 4. Cleanup temp files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # 5. Process and return
        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        if response_json.get('success'):
            server_stats['successful_requests'] += 1
            return jsonify({
                'success': True,
                'your_data': 'processed',
                'processing_time': round(processing_time, 3)
            })
        else:
            server_stats['failed_requests'] += 1
            return jsonify({'error': response_json.get('error')}), 500

    except Exception as e:
        server_stats['failed_requests'] += 1
        logger.error(f"[YOUR_ENDPOINT ERROR] {e}")

        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return jsonify({'error': str(e)}), 500
```

## Endpoint Flow

All new REST endpoints follow this pattern:

1. **Receive multipart file** from HTTP request
2. **Save to /home/n8n/in/** (temporary)
3. **Call /ocr endpoint** internally (Paco's logic)
4. **Cleanup temp files**
5. **Format response** for REST API
6. **Update statistics**

This ensures we leverage 100% of Paco's processing without duplicating code.

## Output Formats

The `/process` endpoint accepts a `format` parameter but currently both modes return the same result.

### Normal (default)
Plain text extraction. The OCR text is returned as-is.

```bash
curl -X POST http://localhost:8503/process \
  -F "file=@factura.pdf" \
  -F "format=normal"
```

### Layout (PENDING - returns same as Normal)
**Status:** Not yet implemented. Currently returns the same result as Normal.

**Why?** The basic text post-processing approach was tested but didn't improve results because:
1. The OCR doesn't preserve X,Y coordinates through the pipeline
2. Text columns get separated (concepts in one block, prices in another)
3. Post-processing can't reconstruct spatial relationships without coordinates

**Future:** Will be implemented with PP-Structure in the experimental repository, using bounding box coordinates for real layout reconstruction.

## Testing

### Test Dashboard
```bash
curl http://localhost:8503/
```

### Test Health
```bash
curl http://localhost:8503/health | jq
```

### Test Stats
```bash
curl http://localhost:8503/stats | jq
```

### Test Process - Normal format
```bash
curl -X POST http://localhost:8503/process \
  -F "file=@test.pdf" \
  -F "format=normal"
```

### Test Process - Layout format (for invoices)
```bash
curl -X POST http://localhost:8503/process \
  -F "file=@factura.pdf" \
  -F "format=layout"
```

### Test Analyze
```bash
curl -X POST http://localhost:8503/analyze \
  -F "file=@test.pdf" | jq -r '.ultra_analysis'
```

### Test Original /ocr (n8n)
```bash
curl -X POST http://localhost:8503/ocr \
  -F "filename=/home/n8n/in/test.pdf"
```

## Docker Commands

```bash
# Build
docker-compose build

# Start
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Common Issues

### Issue: Dashboard not loading
**Solution**: Check Flask is running on port 8503
```bash
docker-compose ps
curl http://localhost:8503/health
```

### Issue: /process endpoint returns 500
**Solution**: Check temp file permissions
```bash
docker exec -it <container> ls -la /home/n8n/in/
```

### Issue: Statistics not updating
**Solution**: Verify `server_stats` is global
- Check line 1780 has `server_stats` dict
- Check each endpoint uses `global server_stats`

## Development Workflow

1. **Never modify Paco's base** (lines 1-1773)
2. **Add new endpoints** after line 2307
3. **Update dashboard** HTML in `dashboard()` function
4. **Test with curl** before committing
5. **Update README.md** with new endpoints
6. **Verify n8n compatibility** - original `/ocr` must work

## Performance

- **Dashboard**: < 100ms (cached stats)
- **/health**: < 50ms (simple check)
- **/stats**: < 100ms (JSON serialization)
- **/process**: ~1-2s (includes Paco's full pipeline)
- **/analyze**: ~1-2s (includes Paco's full pipeline)
- **/ocr**: ~1-2s (Paco's original)

## Code Style

- Use `logger.info(f"[ENDPOINT] message")` for logging
- Use `global server_stats` at start of endpoints
- Clean temp files in `finally` blocks
- Return processing_time in all responses
- Use `try/except` for error handling

## Important Variables

### From Paco's Base (lines 1-1773)
- `doc_preprocessor` - Orientation classification model
- `ocr_instance` - PaddleOCR instance
- `ocr_initialized` - Boolean flag
- `OPENCV_CONFIG` - OpenCV parameters
- `ROTATION_CONFIG` - Rotation parameters

### From API Layer (lines 1774-2312)
- `server_stats` - Statistics dictionary
  - `startup_time`
  - `total_requests`
  - `successful_requests`
  - `failed_requests`
  - `total_processing_time`

## Project Philosophy

**This project is a THIN API LAYER over Paco's proven processing logic.**

- ✅ Add endpoints that call `/ocr` internally
- ✅ Add monitoring and statistics
- ✅ Add dashboard and visualization
- ❌ Don't duplicate preprocessing logic
- ❌ Don't modify Paco's functions
- ❌ Don't break n8n compatibility

---

**Questions?** Read README.md or check Paco's original documentation.
