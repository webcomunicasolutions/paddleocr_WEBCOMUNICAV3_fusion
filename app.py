#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddlePaddle CPU Document Preprocessor
Prepara documentos para OCR con deteccion de orientacion y correccion

LAZY LOADING: Flask arranca primero, modelos se cargan después
DEBUG MODE: Logging extensivo para diagnosticar problemas de arranque
"""

import os
import sys
import json
import subprocess
import logging
import time
import math
import tempfile
import threading
import signal
import traceback
from pathlib import Path
from flask import Flask, request, jsonify

# Configurar logging ANTES de cualquier otra cosa
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Función para obtener uso de memoria
def get_memory_usage():
    """Retorna uso de memoria en MB"""
    try:
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        return f"{mem:.1f}MB"
    except:
        try:
            # Alternativa: leer /proc/self/status
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        mem_kb = int(line.split()[1])
                        return f"{mem_kb/1024:.1f}MB"
        except:
            return "N/A"
    return "N/A"

# Manejador de señales para debug
def signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    logger.error(f"[SIGNAL] Recibida señal {sig_name} ({signum})")
    logger.error(f"[SIGNAL] Stack trace:\n{traceback.format_stack(frame)}")
    sys.exit(1)

# Registrar manejadores de señales
for sig in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(sig, signal_handler)

logger.info(f"[STARTUP] ========== INICIANDO PADDLEOCR V3 ==========")
logger.info(f"[STARTUP] Memoria inicial: {get_memory_usage()}")
logger.info(f"[STARTUP] Python: {sys.version}")
logger.info(f"[STARTUP] PID: {os.getpid()}")

# CONFIGURAR DIRECTORIOS PADDLE ANTES DE IMPORTAR
os.environ['PADDLE_HOME'] = '/home/n8n/.paddleocr'
os.environ['PADDLEX_HOME'] = '/home/n8n/.paddlex'
os.environ['HOME'] = '/home/n8n'

logger.info("[STARTUP] Variables de entorno configuradas")
logger.info(f"[STARTUP] PADDLE_HOME: {os.environ.get('PADDLE_HOME')}")
logger.info(f"[STARTUP] PADDLEX_HOME: {os.environ.get('PADDLEX_HOME')}")

# Imports básicos (rápidos)
logger.info("[STARTUP] Importando OpenCV...")
import cv2
logger.info(f"[STARTUP] OpenCV {cv2.__version__} OK - Memoria: {get_memory_usage()}")

logger.info("[STARTUP] Importando NumPy...")
import numpy as np
logger.info(f"[STARTUP] NumPy {np.__version__} OK - Memoria: {get_memory_usage()}")

# Flask arranca PRIMERO
logger.info("[STARTUP] Creando Flask app...")
app = Flask(__name__)
logger.info(f"[STARTUP] Flask app creada - Memoria: {get_memory_usage()}")

# Variables globales para lazy loading
paddle = None
paddleocr = None
DocImgOrientationClassification = None
models_loaded = False
models_loading = False
models_error = None
startup_time = time.time()

def load_models_background():
    """Carga los modelos de PaddleOCR en segundo plano"""
    global paddle, paddleocr, DocImgOrientationClassification, models_loaded, models_loading, models_error

    # Esperar 5 segundos para que Waitress arranque completamente
    logger.info("[MODELS] Esperando 5 segundos antes de cargar modelos...")
    time.sleep(5)

    logger.info(f"[MODELS] Memoria antes de cargar: {get_memory_usage()}")

    models_loading = True
    logger.info("[MODELS] ========== INICIANDO CARGA DE MODELOS ==========")

    try:
        logger.info("[MODELS] Paso 1/5: Importando paddle...")
        import paddle as _paddle
        paddle = _paddle
        logger.info(f"[MODELS] Paddle {paddle.__version__} importado - Memoria: {get_memory_usage()}")

        logger.info("[MODELS] Paso 2/5: Importando paddleocr...")
        import paddleocr as _paddleocr
        paddleocr = _paddleocr
        logger.info(f"[MODELS] PaddleOCR {paddleocr.__version__} importado - Memoria: {get_memory_usage()}")

        logger.info("[MODELS] Paso 3/5: Importando DocImgOrientationClassification...")
        from paddleocr import DocImgOrientationClassification as _DocImgOrientationClassification
        DocImgOrientationClassification = _DocImgOrientationClassification
        logger.info(f"[MODELS] DocImgOrientationClassification importado - Memoria: {get_memory_usage()}")

        # Ahora inicializar los preprocesadores
        logger.info("[MODELS] Paso 4/5: Inicializando DocPreprocessor...")
        init_docpreprocessor()
        logger.info(f"[MODELS] DocPreprocessor OK - Memoria: {get_memory_usage()}")

        logger.info("[MODELS] Paso 5/5: Inicializando OCR...")
        init_ocr()
        logger.info(f"[MODELS] OCR OK - Memoria: {get_memory_usage()}")

        models_loaded = True
        models_loading = False
        logger.info("[MODELS] ========== TODOS LOS MODELOS CARGADOS ==========")
        logger.info(f"[MODELS] Memoria final: {get_memory_usage()}")

    except Exception as e:
        models_error = str(e)
        models_loading = False
        logger.error(f"[MODELS] ERROR cargando modelos: {e}")
        logger.error(f"[MODELS] Traceback:\n{traceback.format_exc()}")

# NO iniciar el hilo aquí - lo haremos después de que Waitress arranque
logger.info("[STARTUP] Carga de modelos diferida hasta después del arranque del servidor")

# Variables configurables desde ENV
OPENCV_CONFIG = {
    'HSV_LOWER_H': int(os.getenv('OPENCV_HSV_LOWER_H', '0')),
    'HSV_LOWER_S': int(os.getenv('OPENCV_HSV_LOWER_S', '0')),
    'HSV_LOWER_V': int(os.getenv('OPENCV_HSV_LOWER_V', '140')),
    'HSV_UPPER_H': int(os.getenv('OPENCV_HSV_UPPER_H', '180')),
    'HSV_UPPER_S': int(os.getenv('OPENCV_HSV_UPPER_S', '60')),
    'HSV_UPPER_V': int(os.getenv('OPENCV_HSV_UPPER_V', '255')),
    'MIN_AREA_PERCENT': float(os.getenv('OPENCV_MIN_AREA_PERCENT', '0.05')),
    'EPSILON_FACTOR': float(os.getenv('OPENCV_EPSILON_FACTOR', '0.01')),
    'ERODE_ITERATIONS': int(os.getenv('OPENCV_ERODE_ITERATIONS', '1')),
    'DILATE_ITERATIONS': int(os.getenv('OPENCV_DILATE_ITERATIONS', '2')),
    'MIN_WIDTH': int(os.getenv('OPENCV_MIN_WIDTH', '300')),
    'MIN_HEIGHT': int(os.getenv('OPENCV_MIN_HEIGHT', '400')),
    'EROSION_PERCENT': float(os.getenv('OPENCV_EROSION_PERCENT', '0.100')),
    'INNER_SCALE_FACTOR': float(os.getenv('OPENCV_INNER_SCALE_FACTOR', '1.12'))
}

ROTATION_CONFIG = {
    'MIN_CONFIDENCE': float(os.getenv('ROTATION_MIN_CONFIDENCE', '0.7')),
    'MIN_SKEW_ANGLE': float(os.getenv('ROTATION_MIN_SKEW_ANGLE', '0.2'))
}

# Configuracion OCR desde variables de entorno
OCR_CONFIG = {
    'text_det_thresh': float(os.getenv('OCR_TEXT_DET_THRESH', '0.1')),
    'text_det_box_thresh': float(os.getenv('OCR_TEXT_DET_BOX_THRESH', '0.4')),
    'text_det_unclip_ratio': float(os.getenv('OCR_TEXT_DET_UNCLIP_RATIO', '1.5')),
    'text_det_limit_side_len': int(os.getenv('OCR_TEXT_DET_LIMIT_SIDE_LEN', '4800')),
    'text_det_limit_type': os.getenv('OCR_TEXT_DET_LIMIT_TYPE', 'max'),
    'text_recognition_batch_size': int(os.getenv('OCR_TEXT_RECOGNITION_BATCH_SIZE', '6')),
    'textline_orientation_batch_size': int(os.getenv('OCR_TEXTLINE_ORIENTATION_BATCH_SIZE', '1'))
}

# Inicializar DocPreprocessor y OCR globalmente
doc_preprocessor = None
ocr_instance = None
ocr_initialized = False


def init_docpreprocessor():
#    """Verificar versiones de PaddlePaddle e inicializar PP-LCNet_x1_0_doc_ori"""
    """Verificar versiones de PaddlePaddle e inicializar text_image_orientation"""
    global doc_preprocessor

    try:
        # Verificar versiones instaladas
        import paddle
        logger.info(f"[INIT] PaddlePaddle version: {paddle.__version__}")

        import paddleocr
        logger.info(f"[INIT] PaddleOCR version: {paddleocr.__version__}")

        # Verificar si estamos en CPU o GPU
        logger.info(f"[INIT] Paddle device: {paddle.device.get_device()}")
        logger.info(f"[INIT] CUDA available: {paddle.device.cuda.device_count()}")

        logger.info("[INIT] Inicializando DocImgOrientationClassification...")
        from paddleocr import DocImgOrientationClassification
        # Intentar con configuracion especifica para CPU
        doc_preprocessor = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori",
            device="cpu"
        )
        logger.info("[OK] DocImgOrientationClassification inicializado correctamente")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Error inicializando DocImgOrientationClassification: {e}")
        import traceback
        logger.error(f"[ERROR TRACEBACK] {traceback.format_exc()}")
        doc_preprocessor = None
        return False


def init_ocr():
    """Inicializar PaddleOCR con configuracion optimizada desde ENV"""
    global ocr_instance, ocr_initialized

    if ocr_initialized:
        return True

    try:
        logger.info("[OCR INIT] ==========================================================================================")
        logger.info("[OCR INIT]                                Inicializando PaddleOCR                                    ")
        logger.info("[OCR INIT] ==========================================================================================")

        # Verificar versiones
        import paddleocr
        import paddle
        from paddleocr import PaddleOCR
        logger.info(f"[OCR INIT] PaddleOCR version: {paddleocr.__version__}")
        logger.info(f"[OCR INIT] PaddlePaddle version: {paddle.__version__}")
        logger.info(f"[OCR INIT] Device: {paddle.device.get_device()}")

        # Leer configuracion desde ENV
        ocr_config = {
            'ocr_version': os.getenv('OCR_VERSION', 'PP-OCRv3'),
            'lang': os.getenv('OCR_LANG', 'es'),
            'use_doc_orientation_classify': os.getenv('OCR_USE_DOC_ORIENTATION', 'false').lower() == 'true',
            'use_doc_unwarping': os.getenv('OCR_USE_DOC_UNWARPING', 'false').lower() == 'true',
            'use_textline_orientation': os.getenv('OCR_USE_TEXTLINE_ORIENTATION', 'false').lower() == 'true',
            'text_det_thresh': float(os.getenv('OCR_TEXT_DET_THRESH', '0.1')),
            'text_det_box_thresh': float(os.getenv('OCR_TEXT_DET_BOX_THRESH', '0.4')),
            'text_det_limit_side_len': int(os.getenv('OCR_TEXT_DET_LIMIT_SIDE_LEN', '960')),
            'text_det_limit_type': os.getenv('OCR_TEXT_DET_LIMIT_TYPE', 'min'),
            'text_recognition_batch_size': int(os.getenv('OCR_TEXT_RECOGNITION_BATCH_SIZE', '6')),
            'text_det_unclip_ratio': float(os.getenv('OCR_TEXT_DET_UNCLIP_RATIO', '1.5')),
        }

        logger.info("[OCR INIT] Configuracion:")
        logger.info(f"[OCR INIT]   Modelo: {ocr_config['ocr_version']}")
        logger.info(f"[OCR INIT]   Idioma: {ocr_config['lang']}")
        logger.info(f"[OCR INIT]   Deteccion - Umbral: {ocr_config['text_det_thresh']}")
        logger.info(f"[OCR INIT]   Deteccion - Umbral cajas: {ocr_config['text_det_box_thresh']}")
        logger.info(f"[OCR INIT]   Deteccion - Limite lado: {ocr_config['text_det_limit_side_len']}px ({ocr_config['text_det_limit_type']})")
        logger.info(f"[OCR INIT]   Reconocimiento - Batch: {ocr_config['text_recognition_batch_size']}")
        logger.info(f"[OCR INIT]   Orientacion documento: {'SI' if ocr_config['use_doc_orientation_classify'] else 'NO'}")
        logger.info(f"[OCR INIT]   Correccion distorsion: {'SI' if ocr_config['use_doc_unwarping'] else 'NO'}")
        logger.info(f"[OCR INIT]   Orientacion lineas: {'SI' if ocr_config['use_textline_orientation'] else 'NO'}")

        logger.info("[OCR INIT] Cargando modelos...")
        ocr_instance = PaddleOCR(**ocr_config)

        ocr_initialized = True
        logger.info("[OCR INIT] ==========================================================================================")
        logger.info("[OCR INIT] PaddleOCR inicializado correctamente")
        logger.info("[OCR INIT] Modelos cargados en memoria")
        logger.info("[OCR INIT] ==========================================================================================")
        return True

    except Exception as e:
        logger.error(f"[OCR INIT ERROR] Error inicializando PaddleOCR: {e}")
        import traceback
        logger.error(f"[OCR INIT ERROR] {traceback.format_exc()}")
        ocr_instance = None
        ocr_initialized = False
        return False

# LAZY LOADING: No forzar inicializacion al inicio
# Los modelos se cargan en segundo plano via load_models_background()
# init_docpreprocessor() y init_ocr() se llaman cuando se necesiten
logger.info("[START] Inicializacion diferida activada - modelos se cargan en segundo plano")


def find_inner_rectangle(contour, image_shape, config):
    """
    Encuentra el cuadrilátero inscrito dentro del contorno usando erosión morfológica
    para eliminar penínsulas, pero preservando la forma trapezoidal si existe.
    Retorna tanto el trapezoide erosionado como el expandido.
    """
    try:
        # ========================================
        # PASO 1: Crear máscara y aplicar erosión
        # ========================================
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        min_dimension = min(image_shape[0], image_shape[1])
        target_erosion_pixels = int(min_dimension * config['EROSION_PERCENT'])

        kernel_size = max(5, int(target_erosion_pixels / 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        iterations = 3
        actual_erosion = kernel_size * iterations
        actual_percent = (actual_erosion / min_dimension) * 100

        logger.info(f"[IMG] [OCV] [BORDER] Erosion: kernel {kernel_size}x{kernel_size}, {iterations} iter = {actual_erosion}px ({actual_percent:.1f}%)")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask_eroded = cv2.erode(mask, kernel, iterations=iterations)

        # ========================================
        # PASO 2: Encontrar contorno de la máscara erosionada
        # ========================================
        eroded_contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not eroded_contours:
            logger.warning("[IMG] [OCV] [BORDER] No se encontraron contornos después de la erosión")
            return None, None, None, None, None, None

        largest_eroded = max(eroded_contours, key=cv2.contourArea)

        # ========================================
        # PASO 3: Aproximar a 4 puntos (preservar trapezoide)
        # ========================================
        epsilon = config['EPSILON_FACTOR'] * cv2.arcLength(largest_eroded, True)
        approx = cv2.approxPolyDP(largest_eroded, epsilon, True)

        if len(approx) != 4:
            for eps_mult in [0.02, 0.03, 0.04, 0.05, 0.01, 0.06, 0.07]:
                epsilon = eps_mult * cv2.arcLength(largest_eroded, True)
                approx = cv2.approxPolyDP(largest_eroded, epsilon, True)
                if len(approx) == 4:
                    break

        # ========================================
        # PASO 4: Obtener puntos erosionados (azul)
        # ========================================
        if len(approx) == 4:
            eroded_pts = approx.reshape(4, 2).astype("float32")
        else:
            points = np.array(largest_eroded).reshape(-1, 2)
            if len(points) > 4:
                rect = cv2.minAreaRect(points.astype(np.float32))
                eroded_pts = cv2.boxPoints(rect).astype("float32")
            else:
                return None, None, None, None, None, None

        # Ordenar puntos erosionados
        s = eroded_pts.sum(axis=1)
        diff = np.diff(eroded_pts, axis=1).flatten()
        tl = eroded_pts[np.argmin(s)]
        br = eroded_pts[np.argmax(s)]
        tr = eroded_pts[np.argmin(diff)]
        bl = eroded_pts[np.argmax(diff)]
        eroded_pts = np.array([tl, tr, br, bl], dtype="float32")

        # ========================================
        # PASO 5: Expandir para crear puntos finales (verde)
        # ========================================
        expanded_pts = eroded_pts.copy()

        if 'INNER_SCALE_FACTOR' in config and config['INNER_SCALE_FACTOR'] != 1.0:
            scale_factor = config['INNER_SCALE_FACTOR']

            # Calcular cuánto expandir en píxeles
            # Basado en el perímetro promedio del trapecio
            perimeter = (np.linalg.norm(eroded_pts[1] - eroded_pts[0]) + 
                        np.linalg.norm(eroded_pts[2] - eroded_pts[1]) +
                        np.linalg.norm(eroded_pts[3] - eroded_pts[2]) +
                        np.linalg.norm(eroded_pts[0] - eroded_pts[3]))

            # Expansión uniforme: cantidad de píxeles a expandir
            expansion_pixels = (perimeter / 4) * (scale_factor - 1.0)

            # Expandir cada lado perpendicularmente
            expanded_pts = []
            for i in range(4):
                p1 = eroded_pts[i]
                p2 = eroded_pts[(i + 1) % 4]
                p_prev = eroded_pts[(i - 1) % 4]
                p_next = eroded_pts[(i + 2) % 4]

                # Vector del lado actual
                side_vec = p2 - p1
                side_len = np.linalg.norm(side_vec)
                if side_len > 0:
                    side_unit = side_vec / side_len
                else:
                    side_unit = np.array([1, 0])

                # Vector perpendicular hacia afuera (rotación 90° antihoraria)
                perp = np.array([-side_unit[1], side_unit[0]])

                # Vector del lado anterior
                prev_vec = p1 - p_prev
                prev_len = np.linalg.norm(prev_vec)
                if prev_len > 0:
                    prev_unit = prev_vec / prev_len
                else:
                    prev_unit = np.array([1, 0])

                # Vector perpendicular del lado anterior
                prev_perp = np.array([prev_unit[1], prev_unit[0]])

                # Promedio de las perpendiculares para la esquina
                corner_direction = (perp + prev_perp) / 2
                corner_dir_len = np.linalg.norm(corner_direction)
                if corner_dir_len > 0:
                    corner_direction = corner_direction / corner_dir_len

                # Expandir el punto
                if i == 0 or i == 2:
                    # Para puntos 0 y 2, invertir el signo de la componente X de corner_direction
                    corner_direction[0] = -corner_direction[0]

                new_pt = p1 - corner_direction * expansion_pixels
                expanded_pts.append(new_pt)

            expanded_pts = np.array(expanded_pts, dtype="float32")

            logger.info(f"[IMG] [OCV] [BORDER] Expansión paralela aplicada: {scale_factor:.2f} ({(scale_factor-1)*100:.0f}%)")
            logger.info(f"[IMG] [OCV] [BORDER] Píxeles de expansión: {expansion_pixels:.1f}px")

        # ========================================
        # PASO 6: Calcular métricas
        # ========================================
        width_top = np.linalg.norm(expanded_pts[1] - expanded_pts[0])
        width_bottom = np.linalg.norm(expanded_pts[2] - expanded_pts[3])
        height_left = np.linalg.norm(expanded_pts[3] - expanded_pts[0])
        height_right = np.linalg.norm(expanded_pts[2] - expanded_pts[1])

        width_avg = (width_top + width_bottom) / 2
        height_avg = (height_left + height_right) / 2
        aspect_ratio = width_avg / height_avg if height_avg > 0 else 1
        aspect_factor = np.power(aspect_ratio, 1/25) if aspect_ratio > 0 else 1

        # Retornar ambos conjuntos de puntos: erosionados y expandidos
        return expanded_pts, eroded_pts, width_avg, height_avg, aspect_ratio, aspect_factor

    except Exception as e:
        logger.error(f"[IMG] [OCV] [BORDER] Error en find_inner_rectangle: {e}")
        return None, None, None, None, None, None


def det_borders(image_path, npy_file, config):
    """
    Detectar contorno del papel y guardar puntos con visualización de tres niveles:
    - Rojo: contorno original
    - Azul: erosionado (sin penínsulas)
    - Verde: expandido final
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error("FALLO: No se pudo leer la imagen")
            return False, None

        visualization = image.copy()
        original_area = image.shape[0] * image.shape[1]
        logger.info(f"[IMG] [OCV] [BORDER] Imagen original: {image.shape[1]}x{image.shape[0]} pixels")

        # Convertir a HSV y crear máscara
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV_LOWER = np.array([config['HSV_LOWER_H'], config['HSV_LOWER_S'], config['HSV_LOWER_V']])
        HSV_UPPER = np.array([config['HSV_UPPER_H'], config['HSV_UPPER_S'], config['HSV_UPPER_V']])
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=config['ERODE_ITERATIONS'])
        mask = cv2.dilate(mask, kernel, iterations=config['DILATE_ITERATIONS'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (200, 200, 200), 2)
        logger.info(f"[IMG] [OCV] [BORDER] Total contornos encontrados: {len(contours)}")

        if contours:
            largest = max(contours, key=cv2.contourArea)
            detected_area = cv2.contourArea(largest)
            area_percent = (detected_area / original_area) * 100
            logger.info(f"[IMG] [OCV] [BORDER] Contorno mas grande: {area_percent:.1f}% del area total")

            # Dibujar contorno original en amarillo
            cv2.drawContours(visualization, [largest], -1, (0, 255, 255), 3)

            min_area = original_area * config['MIN_AREA_PERCENT']
            if detected_area > min_area:
                # ========================================
                # TRAPEZOIDE ROJO (original)
                # ========================================
                epsilon = config['EPSILON_FACTOR'] * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)

                if len(approx) == 4:
                    outer_pts = approx.reshape(4, 2).astype("float32")
                else:
                    rect = cv2.minAreaRect(largest)
                    outer_pts = cv2.boxPoints(rect).astype("float32")

                # Ordenar y dibujar en rojo
                s = outer_pts.sum(axis=1)
                diff = np.diff(outer_pts, axis=1).flatten()
                tl = outer_pts[np.argmin(s)]
                br = outer_pts[np.argmax(s)]
                tr = outer_pts[np.argmin(diff)]
                bl = outer_pts[np.argmax(diff)]
                outer_pts = np.array([tl, tr, br, bl], dtype="float32")

                outer_pts_int = outer_pts.astype(int)
                cv2.polylines(visualization, [outer_pts_int], True, (0, 0, 255), 2)

                # ========================================
                # TRAPEZOIDES AZUL Y VERDE (erosionado y expandido)
                # ========================================
                expanded_pts, eroded_pts, width_side_in, height_side_in, aspect_ratio_in, aspect_factor_in = find_inner_rectangle(
                    largest, image.shape, config
                )

                if expanded_pts is not None:
                    # Dibujar trapezoide erosionado en AZUL
                    eroded_pts_int = eroded_pts.astype(int)
                    cv2.polylines(visualization, [eroded_pts_int], True, (255, 0, 0), 3)  # Azul

                    # Dibujar trapezoide expandido en VERDE
                    expanded_pts_int = expanded_pts.astype(int)
                    cv2.polylines(visualization, [expanded_pts_int], True, (0, 255, 0), 4)  # Verde

                    # Marcar vértices del verde (final)
                    for i, pt in enumerate(expanded_pts_int):
                        cv2.circle(visualization, tuple(pt), 8, (0, 255, 0), -1)
                        cv2.circle(visualization, tuple(pt), 10, (255, 255, 255), 2)
                        cv2.putText(visualization, str(i), tuple(pt + [15, -10]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Calcular areas
                    green_area = cv2.contourArea(expanded_pts)
                    red_area = cv2.contourArea(outer_pts)

                    # Usar el menor
                    if red_area < green_area:
                        pts_final = outer_pts
                        logger.info(f"[IMG] [OCV] [BORDER] Usando trapezoide ROJO (más pequeño): {red_area:.0f} < {green_area:.0f}")
                    else:
                        pts_final = expanded_pts
                        logger.info(f"[IMG] [OCV] [BORDER] Usando trapezoide VERDE (más pequeño): {green_area:.0f} <= {red_area:.0f}")

                    detection_method = "eroded-expanded"

                else:
                    # Fallback
                    logger.warning("[IMG] [OCV] [BORDER] Fallback: usando contorno reducido")
                    center = np.mean(outer_pts, axis=0)
                    pts_final = []
                    for pt in outer_pts:
                        new_pt = center + (pt - center) * 0.9
                        pts_final.append(new_pt)
                    pts_final = np.array(pts_final, dtype=np.float32)

                    pts_int = pts_final.astype(int)
                    cv2.polylines(visualization, [pts_int], True, (0, 255, 0), 4)
                    detection_method = "fallback"

                # Calcular ángulo
                dx = pts_final[1][0] - pts_final[0][0]
                dy = pts_final[1][1] - pts_final[0][1]
                paper_angle = math.degrees(math.atan2(dy, dx))
                if paper_angle < 0:
                    paper_angle += 360

                # Añadir leyenda
                cv2.putText(visualization, "Metodo: " + detection_method, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(visualization, "Rojo: Original", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(visualization, "Azul: Erosionado", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(visualization, "Verde: Final", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(visualization, f"Area: {area_percent:.1f}%", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(visualization, f"Angulo: {paper_angle:.1f} deg", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Guardar visualización
                output_base = npy_file.replace('.npy', '')
                vis_filename = f"{output_base}.png"
                cv2.imwrite(vis_filename, visualization)
                logger.info(f"[IMG] [OCV] [BORDER] Imagen provisional: {vis_filename}")

                # Guardar puntos finales
                np.save(npy_file, pts_final)
                logger.info(f"[IMG] [OCV] [BORDER] Puntos guardados en {npy_file}")

                return True, f"{detection_method}|{area_percent:.1f}%|{paper_angle:.1f}deg"

            else:
                logger.error(f"FALLO: Area muy pequena ({area_percent:.1f}%)")
                return False, None
        else:
            logger.error("FALLO: No se encontraron contornos")
            return False, None

    except Exception as e:
        logger.error(f"FALLO: {e}")
        return False, None


def fix_perspective(image_path, npy_file, perspective_file, config):
    """
    Corregir perspectiva aplicando factor de aspecto para compensar
    la expansión diferencial en dimensiones
    """
    try:
        image = cv2.imread(image_path)
        pts = np.load(npy_file)

        logger.info(f"[IMG] [OCV] [PERSPECTIVE] Aplicando correccion perspectiva")

        # Ordenar puntos
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        src = np.array([tl, tr, br, bl], dtype="float32")

        # Calcular dimensiones base
        width_base = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height_base = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

        # Aplicar compensación por aspecto si está configurado
        if 'ASPECT_COMPENSATION' in config and config['ASPECT_COMPENSATION']:
            aspect_ratio = width_base / height_base if height_base > 0 else 1
            aspect_factor = np.power(aspect_ratio, 1/25)

            # Ajustar dimensiones con el factor de aspecto
            # Nota: Aquí aplicamos la compensación inversa porque ya se aplicó en la expansión
            width = int(width_base / aspect_factor)
            height = int(height_base * aspect_factor)
            
            logger.info(f"[IMG] [OCV] [PERSPECTIVE] Compensación de aspecto aplicada: {aspect_factor:.3f}")
            logger.info(f"[IMG] [OCV] [PERSPECTIVE] Dimensiones: {width_base}x{height_base} -> {width}x{height}")
        else:
            width = width_base
            height = height_base
        
        # Aplicar límites mínimos
        width = max(width, config.get('MIN_WIDTH', 100))
        height = max(height, config.get('MIN_HEIGHT', 100))
        
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(image, M, (width, height), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
        
        cv2.imwrite(perspective_file, corrected, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        return True, f"{width}x{height}"
        
    except Exception as e:
        logger.error(f"FALLO: {e}")
        return False, None


def fix_orientation(img_path, doc_preprocessor):
    """
    Detectar y corregir orientacion de imagen
    Returns: (success, orientation_degrees, confidence, rotated)
    """
    try:
        if not doc_preprocessor:
            logger.info("[IMG] [PADDLE] [ORIENTATION] Modelo no disponible")
            return False, 0, 0.0, False

        output = doc_preprocessor.predict(img_path, batch_size=1)
        orientation = '0'
        confidence = 0.0

        for res in output:
            result_data = res.res if hasattr(res, 'res') else res
            if isinstance(result_data, dict):
                label_names = result_data.get('label_names', [])
                scores = result_data.get('scores', [])
                if label_names and scores:
                    orientation = label_names[0]
                    confidence = scores[0]

        # Rotar si es necesario
        rotated = False
        if orientation in ['90', '180', '270'] and confidence > ROTATION_CONFIG['MIN_CONFIDENCE']:
            img = cv2.imread(img_path)
            if orientation == '90':
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation == '180':
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif orientation == '270':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(img_path, img)
            rotated = True

        return True, int(orientation), confidence, rotated

    except Exception as e:
        logger.warning(f"[IMG] [PADDLE] [ORIENTATION] Error detectando orientacion: {e}")
        return False, 0, 0.0, False


def fix_deskew(img_path):
    """
    Detectar y corregir inclinacion de imagen usando ImageMagick
    Returns: (success, skew_angle, corrected)
    """
    try:
        # Detectar angulo de inclinacion
        result = subprocess.run(['convert', img_path, '-deskew', '45%', '-format', '%[deskew:angle]', 'info:'],
                              capture_output=True, text=True)

        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("[IMG] [CONVERT] [DESKEW] Error detectando inclinacion")
            return False, 0.0, False

        skew_angle = result.stdout.strip()

        try:
            skew_float = float(skew_angle)
            skew_abs = abs(skew_float)

            corrected = False
            if skew_abs > ROTATION_CONFIG['MIN_SKEW_ANGLE']:
                deskewed_path = img_path.replace('.png', '_deskewed.png')
                result = subprocess.run([
                    'convert', img_path,
                    '-background', 'white',
                    '-interpolate', 'bicubic',
                    '-deskew', '45%',
                    '-fuzz', '10%',
                    '+repage',
                    deskewed_path
                ], capture_output=True, text=True)

                if result.returncode == 0 and os.path.exists(deskewed_path):
                    subprocess.run(['mv', deskewed_path, img_path])
                    corrected = True
                else:
                    logger.warning("[IMG] [CONVERT] [DESKEW] Error aplicando correccion")
                    return False, skew_float, False

            return True, skew_float, corrected

        except ValueError:
            logger.warning(f"[IMG] [CONVERT] [DESKEW] No se pudo parsear angulo: {skew_angle}")
            return False, 0.0, False

    except Exception as e:
        logger.warning(f"[IMG] [CONVERT] [DESKEW] Error procesando inclinacion: {e}")
        return False, 0.0, False


def init_pdf_prep(n8nHomeDir, base_name, ext):
    """Preparacion inicial de PDF - desproteger y copiar"""
    try:
        filename = f"{base_name}{ext}"
        in_file = f"{n8nHomeDir}/in/{filename}"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"

        logger.info(f"[PDF] Preparando PDF: {in_file}")

        # Leer configuracion del JSON
        json_file = f"{n8nHomeDir}/json/{filename}.json"
        empresaNif = ""

        if os.path.exists(json_file):
            try:
                result = subprocess.run(['jq', '-r', '.empresaNif // ""', json_file], capture_output=True, text=True)
                if result.returncode == 0:
                    empresaNif = result.stdout.strip()
                    logger.info(f"[JSON] empresaNif: {empresaNif}")
            except Exception as e:
                logger.warning(f"[JSON] Error leyendo JSON: {e}")

        # Verificar si esta protegido
        result = subprocess.run(['pdfinfo', in_file], capture_output=True, text=True)

        if 'Incorrect password' in result.stderr and empresaNif:
            logger.info("[PDF] PDF protegido, desprotegiendo...")

            # Desproteger con empresaNif
            tmp_file = f"{in_file}_unlocked.pdf"
            result = subprocess.run([
                'qpdf', '--password=' + empresaNif, '--decrypt',
                in_file, tmp_file
            ], capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(tmp_file):
                # Mover archivo desprotegido
                subprocess.run(['mv', tmp_file, in_file])
                logger.info("[PDF] PDF desprotegido correctamente")
            else:
                logger.warning("[PDF] No se pudo desproteger PDF")

        # Copiar a directorio OCR
        subprocess.run(['cp', in_file, out_pdf])
        logger.info(f"[PDF] PDF copiado a {out_pdf}")

        return True

    except Exception as e:
        logger.error(f"[PDF ERROR] {e}")
        return False


def init_img_prep(n8nHomeDir, base_name, ext):
    """Preparacion inicial de imagen - perspectiva y crear PDF"""
    try:
        filename = f"{base_name}{ext}"
        in_file = f"{n8nHomeDir}/in/{filename}"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"

        logger.info(f"[IMG] Preparando imagen: {in_file}")

        # Rutas para archivos intermedios
        npy_file = f"{n8nHomeDir}/ocr/{base_name}_1.1.borders.npy"
        ocv_img = f"{n8nHomeDir}/ocr/{base_name}_1.2.ocv.png"
        fallback_pdf_file = f"{n8nHomeDir}/ocr/{base_name}_1.4.ocv.pdf"

        # 1.1. Detectar bordes/contorno
        success, detect_result = det_borders(in_file, npy_file, OPENCV_CONFIG)
        if success:
            logger.info(f"[IMG] [OCV] [BORDER] Resultado OK - {detect_result}")
        else:
            logger.warning(f"[IMG] [OCV] [BORDER] Fallo en deteccion de bordes")

        # 1.2. Corregir perspectiva (solo si 1.1. funciono)
        if os.path.exists(npy_file):
            success, perspective_result = fix_perspective(in_file, npy_file, ocv_img, OPENCV_CONFIG)
            if success:
                logger.info(f"[IMG] [OCV] [PERSPECTIVE] Resultado OK - {perspective_result} pixels")
            else:
                logger.warning("[IMG] [OCV] [PERSPECTIVE] Fallo en correccion de perspectiva")

        # 1.3. Crear PDF preocr
        if os.path.exists(ocv_img):
            result = subprocess.run(['convert', ocv_img, '-compress', 'jpeg', '-quality', '75', out_pdf], capture_output=True, text=True)

            if result.returncode == 0:
                # Mostrar resumen
                final_size_result = subprocess.run(['identify', '-format', '%wx%h', ocv_img], capture_output=True, text=True)

                if final_size_result.returncode == 0:
                    final_size = final_size_result.stdout.strip()
                    logger.info(f"[IMG] [PDF] PDF creado con imagen procesada: {final_size} pixels")
            else:
                logger.error("[IMG] [PDF] Fallo al crear PDF con imagen procesada")

        # 1.3.1. Fallback: crear PDF con imagen original si no existe
        if not os.path.exists(out_pdf):
            result = subprocess.run(['convert', in_file, '-compress', 'jpeg', '-quality', '75', out_pdf], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("[IMG] [PDF] PDF creado con imagen original")
            else:
                logger.error("[IMG] [PDF] Fallo al crear PDF con imagen original")
                return False

        return True

    except Exception as e:
        logger.error(f"[IMG ERROR] {e}")
        return False


def det_scanned(pdf_path, page_num=1):
    """
    Detectar si una pagina especifica es escaneada o vectorial

    Criterios (OR):
    - Es vectorial si tiene fuentes embebidas
    - Es vectorial si NO tiene ninguna imagen >80% del area de pagina

    Returns: True si es escaneada, False si es vectorial
    """
    try:
        import subprocess
        import fitz  # PyMuPDF

        # 1. Verificar fuentes embebidas
        result = subprocess.run(
            ['pdffonts', '-f', str(page_num), '-l', str(page_num), pdf_path], 
            capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.warning(f"[det_scanned] pdffonts fallo en pagina {page_num}")
            return True  # Asumir escaneada si hay error

        # Contar fuentes embebidas
        embedded_fonts = 0
        lines = result.stdout.splitlines()

        for line in lines[2:]:  # Saltar headers
            if line.strip():
                parts = line.split()
                if len(parts) >= 5 and parts[4] == 'yes':  # columna 'emb'
                    embedded_fonts += 1

        # Si hay fuentes embebidas, es vectorial
        if embedded_fonts > 0:
            logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL ({embedded_fonts} fuentes embebidas)")
            return False

        # 2. Si no hay fuentes embebidas, verificar imagenes con PyMuPDF
        try:
            pdf = fitz.open(pdf_path)

            # Verificar que la pagina existe
            if page_num > len(pdf):
                logger.warning(f"[det_scanned] Pagina {page_num} no existe")
                return False

            page = pdf[page_num - 1]  # PyMuPDF usa indice base 0

            # Obtener area de la pagina
            page_width = page.rect.width
            page_height = page.rect.height
            page_area = page_width * page_height
            threshold_percentage = 80.0

            logger.info(f"[DET_SCANNED] Pagina: {page_width:.0f}x{page_height:.0f} pts")

            # Obtener todas las imagenes de la pagina
            images = page.get_images(full=True)

            if not images:
                # Sin imagenes = probablemente vectorial puro
                logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL (sin imagenes, {embedded_fonts} fuentes embebidas)")
                pdf.close()
                return False

            # Verificar el tamano de cada imagen en la pagina
            has_large_image = False

            for img_index, img_info in enumerate(images):
                xref = img_info[0]

                # Obtener los rectangulos donde aparece esta imagen
                try:
                    img_rects = page.get_image_rects(xref)

                    for rect in img_rects:
                        # Calcular area de la imagen
                        img_area = rect.width * rect.height
                        percentage = (img_area / page_area) * 100

                        logger.debug(f"[DET_SCANNED] Imagen {img_index}: {rect.width:.0f}x{rect.height:.0f} pts = {percentage:.1f}% del area")

                        if percentage > threshold_percentage:
                            logger.info(f"[DET_SCANNED] Imagen grande detectada: {percentage:.1f}% del area de pagina")
                            has_large_image = True
                            break

                    if has_large_image:
                        break

                except Exception as e:
                    logger.debug(f"[DET_SCANNED] Error obteniendo rectangulos de imagen {img_index}: {e}")
                    continue

            pdf.close()

            # Determinar resultado
            if has_large_image:
                logger.info(f"[DET_SCANNED] Detectada pagina ESCANEADA (imagen >{threshold_percentage:.0f}% del area)")
                return True
            else:
                logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL (sin imagenes grandes, {embedded_fonts} fuentes embebidas)")
                return False

        except Exception as e:
            logger.warning(f"[DET_SCANNED] Error usando PyMuPDF: {e}")
            # Fallback: si no podemos verificar imagenes, asumir vectorial si no hay fuentes embebidas
            return False

    except Exception as e:
        logger.error(f"[DET_SCANNED] Error en pagina {page_num}: {e}")
        return True  # Asumir escaneada en caso de error


def extract_pdf_images(n8nHomeDir, base_name, in_pdf, out_png, target_dpi=288):
    """
    Extraer imagenes de PDF vectorial y crear PNG con solo imagenes posicionadas

    Args:
        n8nHomeDir: Directorio base de n8n
        base_name: Nombre base del archivo
        in_pdf: Path del PDF de entrada
        out_png: Path del PNG de salida
        target_dpi: DPI objetivo para el PNG de salida (default: 144)
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import io

        logger.info(f"[EXTRACT_IMAGES] Extrayendo imagenes de: {in_pdf}")
        logger.info(f"[EXTRACT_IMAGES] DPI objetivo: {target_dpi}")

        # Calcular factor de escala respecto a 72 DPI (base de PDF)
        scale_factor = target_dpi / 72.0

        # Abrir PDF con PyMuPDF
        pdf_document = fitz.open(in_pdf)

        # Procesar primera pagina (PDF individual)
        page = pdf_document[0]

        # Obtener dimensiones de la pagina en puntos
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        logger.info(f"[EXTRACT_IMAGES] Pagina original: {page_width:.1f}x{page_height:.1f} pts (72 DPI)")

        # Calcular dimensiones escaladas para el canvas
        canvas_width = int(page_width * scale_factor)
        canvas_height = int(page_height * scale_factor)

        logger.info(f"[EXTRACT_IMAGES] Canvas escalado: {canvas_width}x{canvas_height} px ({target_dpi} DPI)")

        # Obtener lista de imagenes en la pagina
        image_list = page.get_images(full=True)

        logger.info(f"[EXTRACT_IMAGES] Imagenes encontradas: {len(image_list)}")

        if not image_list:
            logger.warning(f"[EXTRACT_IMAGES] Sin imagenes en pagina, creando PNG vacio")
            # Crear PNG vacio con dimensiones escaladas
            empty_img = Image.new('RGB', (canvas_width, canvas_height), 'white')
            empty_img.save(out_png, dpi=(target_dpi, target_dpi))
            pdf_document.close()
            return

        # Crear imagen base con dimensiones escaladas (fondo blanco)
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # Contador de imagenes procesadas exitosamente
        images_processed = 0

        # Procesar cada imagen
        for img_index, img_info in enumerate(image_list):
            try:
                # img_info contiene: [xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter]
                xref = img_info[0]

                # Extraer la imagen
                try:
                    img_dict = pdf_document.extract_image(xref)
                    if not img_dict or "image" not in img_dict:
                        logger.warning(f"[EXTRACT_IMAGES] No se pudo extraer imagen {img_index+1} (xref={xref})")
                        continue

                    # Obtener datos de la imagen
                    img_data = img_dict["image"]
                    img_ext = img_dict["ext"]
                    orig_width = img_dict["width"]
                    orig_height = img_dict["height"]

                    # Crear PIL Image desde los bytes
                    pil_img = Image.open(io.BytesIO(img_data))

                    logger.debug(f"[EXTRACT_IMAGES] Imagen {img_index+1} extraida: {orig_width}x{orig_height} {img_ext}")

                except Exception as e:
                    logger.warning(f"[EXTRACT_IMAGES] Error extrayendo imagen {img_index+1}: {e}")
                    continue

                # Obtener las posiciones de esta imagen en la pagina usando get_image_rects
                try:
                    img_rects = page.get_image_rects(xref)

                    if not img_rects:
                        logger.warning(f"[EXTRACT_IMAGES] No se encontraron posiciones para imagen {img_index+1}")
                        continue

                    # Procesar cada instancia de la imagen (puede aparecer varias veces)
                    for inst_index, rect in enumerate(img_rects):
                        # rect es un fitz.Rect con coordenadas en puntos PDF (72 DPI)
                        x0 = rect.x0
                        y0 = rect.y0
                        rect_width = rect.width
                        rect_height = rect.height

                        # Escalar coordenadas y dimensiones segun el DPI objetivo
                        x_pos = int(x0 * scale_factor)
                        y_pos = int(y0 * scale_factor)
                        target_width = int(rect_width * scale_factor)
                        target_height = int(rect_height * scale_factor)

                        logger.debug(f"[EXTRACT_IMAGES] Imagen {img_index+1}.{inst_index+1}:")
                        logger.debug(f"[EXTRACT_IMAGES]   - Original en PDF: {int(rect_width)}x{int(rect_height)} en ({int(x0)}, {int(y0)}) pts")
                        logger.debug(f"[EXTRACT_IMAGES]   - Escalada a {target_dpi} DPI: {target_width}x{target_height} en ({x_pos}, {y_pos}) px")

                        # Validar dimensiones
                        if target_width <= 0 or target_height <= 0:
                            logger.warning(f"[EXTRACT_IMAGES] Dimensiones invalidas: {target_width}x{target_height}")
                            continue

                        # Redimensionar imagen al tamano escalado
                        try:
                            # Usar LANCZOS para mejor calidad al escalar
                            pil_img_resized = pil_img.resize((target_width, target_height), Image.LANCZOS)
                            logger.debug(f"[EXTRACT_IMAGES] Redimensionada de {orig_width}x{orig_height} a {target_width}x{target_height}")
                        except Exception as resize_err:
                            logger.warning(f"[EXTRACT_IMAGES] Error redimensionando: {resize_err}")
                            continue

                        # Verificar que la imagen cabe en el canvas escalado
                        if (x_pos + target_width > canvas_width) or (y_pos + target_height > canvas_height):
                            logger.warning(f"[EXTRACT_IMAGES] Imagen excede limites del canvas, ajustando")
                            # Ajustar si es necesario
                            if x_pos + target_width > canvas_width:
                                crop_width = canvas_width - x_pos
                                if crop_width > 0:
                                    pil_img_resized = pil_img_resized.crop((0, 0, crop_width, target_height))
                                    target_width = crop_width
                            if y_pos + target_height > canvas_height:
                                crop_height = canvas_height - y_pos
                                if crop_height > 0:
                                    pil_img_resized = pil_img_resized.crop((0, 0, target_width, crop_height))
                                    target_height = crop_height

                        # Pegar en canvas escalado
                        # Las coordenadas Y en PyMuPDF ya tienen origen arriba-izquierda (correcto para PIL)
                        canvas.paste(pil_img_resized, (x_pos, y_pos))
                        images_processed += 1

                        logger.debug(f"[EXTRACT_IMAGES] Imagen pegada en canvas en posicion ({x_pos}, {y_pos})")

                except Exception as e:
                    logger.warning(f"[EXTRACT_IMAGES] Error obteniendo posiciones de imagen {img_index+1}: {e}")
                    continue

            except Exception as e:
                logger.warning(f"[EXTRACT_IMAGES] Error procesando imagen {img_index+1}: {e}")
                continue

        # Informar resultado
        if images_processed == 0:
            logger.warning(f"[EXTRACT_IMAGES] No se pudo procesar ninguna imagen, creando PNG vacio")
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        else:
            logger.info(f"[EXTRACT_IMAGES] Imagenes procesadas exitosamente: {images_processed}")

        # Guardar PNG resultante con metadatos DPI
        canvas.save(out_png, dpi=(target_dpi, target_dpi))
        logger.info(f"[EXTRACT_IMAGES] PNG creado: {out_png} a {target_dpi} DPI")

        # Cerrar documento PDF
        pdf_document.close()

    except Exception as e:
        logger.error(f"[EXTRACT_IMAGES ERROR] Error extrayendo imagenes: {e}")
        # Crear PNG vacio como fallback con DPI por defecto
        from PIL import Image
        fallback_width = int(595 * (target_dpi / 72.0))
        fallback_height = int(842 * (target_dpi / 72.0))
        fallback_img = Image.new('RGB', (fallback_width, fallback_height), 'white')
        fallback_img.save(out_png, dpi=(target_dpi, target_dpi))


def create_spdf(n8nHomeDir, base_name, in_pdf, spdf, page_num, ocr_dpi=288):
    """
    Procesar una pagina individual: escaneada o vectorial
    Genera PDF perfecto de una sola pagina
    """
    logger.info(f"[CREATE_SPDF] Procesando: {in_pdf}")

    page_start_time = time.time()

    # Detectar tipo de pagina
    page_scanned = det_scanned(in_pdf)

    if page_scanned:

        # Extraer a imagen con ocr_dpi
        subprocess.run(['pdftoppm', '-png', '-f', '1', '-l', '1', '-r', str(ocr_dpi), in_pdf, in_pdf.replace('.pdf', '')], check=True)

        # Detectar y corregir orientacion
        in_png = f"{n8nHomeDir}/ocr/{base_name}_2.2.page-{page_num}.png"
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.3.page-{page_num}.png"
        subprocess.run(['cp', in_pdf.replace('.pdf', '-1.png'), in_png], check=True)
        subprocess.run(['cp', in_png, out_png], check=True)
        logger.info(f"[ORIENTATION] Detectando orientacion pagina {page_num}...")
        success, degrees, conf, rotated = fix_orientation(out_png, doc_preprocessor)

        if success:
            action = " - CORREGIDO" if rotated else ""
            logger.info(f"[ORIENTATION] Pagina {page_num}: {degrees} grados (confianza: {conf:.3f}){action}")

        # Detectar y corregir inclinacion
        in_png = f"{n8nHomeDir}/ocr/{base_name}_2.3.page-{page_num}.png"
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.4.page-{page_num}.png"
        subprocess.run(['cp', in_png, out_png], check=True)
        logger.info(f"[DESKEW] Detectando inclinacion pagina {page_num}...")
        success, angle, corrected = fix_deskew(out_png)

        if success:
            action = " - CORREGIDO" if corrected else ""
            logger.info(f"[DESKEW] Pagina {page_num}: {angle:.2f} grados{action}")

    else:
        # Extraer imagenes a PNG temporal
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.4.page-{page_num}.png"
        extract_pdf_images(n8nHomeDir, base_name, in_pdf, out_png)

    # Ejecutar OCR sobre la imagen extraida y preparada de la pagina
    logger.info(f"[OCR] Ejecutando OCR en pagina {page_num}...")
    ocr_start = time.time()

    # Reintentos para OCR
    page_ocr_result = None
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            page_ocr_result = ocr_instance.predict(out_png)
            ocr_time = time.time() - ocr_start
            if page_ocr_result and len(page_ocr_result) > 0:
                texts = page_ocr_result[0].get('rec_texts', [])
                scores = page_ocr_result[0].get('rec_scores', [])
                avg_conf = sum(scores)/len(scores) if scores else 0
                logger.info(f"[OCR] Pagina {page_num}: {len(texts)} bloques detectados")
                logger.info(f"[OCR] Confianza promedio: {avg_conf:.3f}")
                logger.info(f"[OCR] Tiempo OCR: {ocr_time:.2f}s")
            else:
                logger.warning(f"[OCR] Pagina {page_num}: Sin texto detectado")
                page_ocr_result = None
            break  # Exito, salir del bucle de reintentos

        except Exception as e:
            logger.error(f"[OCR] Error en pagina {page_num} (intento {attempt}): {e}")
            if attempt < max_attempts:
                logger.info(f"[OCR] Esperando 1 segundo antes del siguiente intento...")
                time.sleep(1)
            else:
                logger.error(f"[OCR] Error definitivo tras {max_attempts} intentos")
                page_ocr_result = None

    # Procesar resultado OCR
    if page_ocr_result and len(page_ocr_result) > 0:
        text_lines, confidences, coordinates = parse_paddleocr_result(page_ocr_result[0])
    else:
        text_lines, confidences, coordinates = [], [], []

    # Crear SPDF con texto buscable
    try:
        if page_scanned:
            # Base: imagen procesada
            compose_pdf_ocr(out_png, (text_lines, confidences, coordinates), spdf, True)
        else:
            # Base: PDF vectorial original
            compose_pdf_ocr(in_pdf, (text_lines, confidences, coordinates), spdf, False)

        logger.info(f"[CREATE_SPDF] PDF con OCR guardado en: {spdf}")

    except Exception as e:
        logger.error(f"[CREATE_SPDF] Error creando PDF final: {e}")
        # Fallback: copiar PDF original
        subprocess.run(['cp', in_pdf, spdf], check=True)
        logger.info(f"[CREATE_SPDF] PDF original copiado como fallback: {spdf}")

    page_time = time.time() - page_start_time
    logger.info(f"[CREATE_SPDF] Pagina {page_num} completada en {page_time:.2f}s: {spdf}")


def proc_pdf_ocr(n8nHomeDir, base_name, ext):
    """Procesar PDF: detectar orientacion, corregir inclinacion y ejecutar OCR"""
    global doc_preprocessor, ocr_instance

    try:
        in_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_3.0.ocr.pdf"
        final_pdf = f"{n8nHomeDir}/pdf/{base_name}{ext}.pdf"

        logger.info("[PROC_PDF_OCR] ==========================================================================================")
        logger.info(f"[PROC_PDF_OCR] Procesando: {in_pdf}")
        logger.info("[PROC_PDF_OCR] ==========================================================================================")

        # Verificar que exista el archivo
        if not os.path.exists(in_pdf):
            logger.error(f"[PROC_PDF_OCR] Archivo no encontrado: {in_pdf}")
            return False, "File not found", None

        # Verificar modelos inicializados
        if not doc_preprocessor:
            logger.info("[PROC_PDF_OCR] Inicializando modelo de orientacion...")
            if not initialize_docpreprocessor():
                logger.warning("[PROC_PDF_OCR] Modelo de orientacion no disponible, continuando sin rotacion")

        if not ocr_instance:
            logger.info("[PROC_PDF_OCR] Inicializando PaddleOCR...")
            if not initialize_ocr():
                logger.error("[PROC_PDF_OCR] No se pudo inicializar PaddleOCR")
                return False, "OCR initialization failed", None

        # Obtener numero de paginas
        result = subprocess.run(['pdfinfo', in_pdf], capture_output=True, text=True)
        pages = 1
        for line in result.stdout.splitlines():
            if "Pages:" in line:
                pages = int(line.split(":")[1].strip())
                break

        # Extraer paginas individuales en /ocr
        subprocess.run(['pdfseparate', in_pdf, f'{n8nHomeDir}/ocr/{base_name}_2.1.page-%d.pdf'], check=True)
        logger.info(f"[PROC_PDF_OCR] Paginas ({pages}): {base_name}_2.1.page-1.pdf - {base_name}_2.1.page-{pages}.pdf")

        # Procesar cada pagina individualmente
        mpdf = []
        total_start_time = time.time()

        for page in range(1, pages + 1):
            page_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.1.page-{page}.pdf"
            spdf = f"{n8nHomeDir}/ocr/{base_name}_2.6.spdf-{page}.pdf"

            logger.info(f"[PROC_PDF_OCR] =================================  Iniciando pagina {page}/{pages}  ===================================")

            # Procesar pagina individual
            create_spdf(n8nHomeDir, base_name, page_pdf, spdf, page)

            # Verificar que se creo correctamente
            if os.path.exists(spdf):
                mpdf.append(spdf)
            else:
                logger.error(f"[PROC_PDF_OCR] Error: No se creo {spdf}")
                return False, f"Failed to create page {page}", None

        # Combinar todas las paginas procesadas
        logger.info(f"[PROC_PDF_OCR] Combinando {len(mpdf)} paginas procesadas...")
        subprocess.run(['pdfunite'] + mpdf + [out_pdf], check=True)
        out_size_kb = os.path.getsize(out_pdf) / 1024
        logger.info(f"[PROC_PDF_OCR] PDF combinado creado ({out_size_kb:.0f}kB): {out_pdf}")

        # Generar en ubicacion final
#        subprocess.run(['cp', out_pdf, final_pdf], check=True)
        subprocess.run([
           'gs',
           '-sDEVICE=pdfwrite',
           '-dCompatibilityLevel=1.4',
           '-dNOPAUSE',
           '-dQUIET',
           '-dBATCH',
           '-dAutoRotatePages=/None',
           '-dColorImageDownsampleType=/Bicubic',
           '-dColorImageResolution=288',
           '-dGrayImageDownsampleType=/Bicubic', 
           '-dGrayImageResolution=288',
           '-dOptimize=true',
           '-dCompressPages=true',
           f'-sOutputFile={final_pdf}',
           out_pdf
        ], check=True)
        final_size_kb = os.path.getsize(final_pdf) / 1024
        logger.info(f"[PROC_PDF_OCR] PDF final creado ({final_size_kb:.0f}kB): {final_pdf}")

        # Calcular estadisticas consolidadas
        total_time = time.time() - total_start_time

        # Leer texto consolidado de todas las paginas (opcional)
        # Esto es para compatibilidad con el sistema anterior
        try:
            # Extraer texto del PDF final para estadisticas
            result_text = subprocess.run(['pdftotext', final_pdf, '-'], capture_output=True, text=True)
            extracted_text = result_text.stdout if result_text.returncode == 0 else ""
            text_length = len(extracted_text.strip())

            logger.info("[PROC_PDF_OCR] ==========================================================================================")
            logger.info(f"[PROC_PDF_OCR] Proceso completado exitosamente")
            logger.info(f"[PROC_PDF_OCR] Total paginas procesadas: {pages}")
            logger.info(f"[PROC_PDF_OCR] Caracteres de texto final: {text_length}")
            logger.info(f"[PROC_PDF_OCR] Tiempo total: {total_time:.2f}s")
            logger.info("[PROC_PDF_OCR] ==========================================================================================")

            return True, "Success", {
                'text_lines': extracted_text.splitlines() if extracted_text else [],
                'confidences': [],  # No disponible en flujo individual
                'total_blocks': len(extracted_text.splitlines()) if extracted_text else 0,
                'pages': pages,
                'processing_time': total_time
            }

        except Exception as text_error:
            logger.warning(f"[PROC_PDF_OCR] Error extrayendo texto final: {text_error}")

            return True, "Success", {
                'text_lines': [],
                'confidences': [],
                'total_blocks': 0,
                'pages': pages,
                'processing_time': total_time
            }

    except Exception as e:
        logger.error(f"[PROC_PDF_OCR ERROR] Error critico: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e), None


def parse_paddleocr_result(ocr_result):
    """Procesar resultado OCR de PaddleOCR v3 (.predict())"""
    text_lines = []
    confidences = []
    coordinates_list = []

    if not ocr_result:
        return text_lines, confidences, coordinates_list

    try:
        logger.info("[OCR PROCESS] Procesando resultado OCR...")

        # PaddleOCR v3 devuelve diccionario con rec_texts, rec_scores, etc.
        if isinstance(ocr_result, dict):
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
            polys = ocr_result.get('rec_polys', [])

            for i, text in enumerate(texts):
                if text and text.strip():
                    text_lines.append(text.strip())
                    confidences.append(scores[i] if i < len(scores) else 0.0)
                    if i < len(polys):
                        coordinates_list.append(polys[i])
                    else:
                        coordinates_list.append([])

        # Si viene en lista (multipagina), procesar cada elemento
        elif isinstance(ocr_result, list):
            for page_result in ocr_result:
                if isinstance(page_result, dict):
                    texts = page_result.get('rec_texts', [])
                    scores = page_result.get('rec_scores', [])
                    polys = page_result.get('rec_polys', [])

                    for i, text in enumerate(texts):
                        if text and text.strip():
                            text_lines.append(text.strip())
                            confidences.append(scores[i] if i < len(scores) else 0.0)
                            if i < len(polys):
                                coordinates_list.append(polys[i])
                            else:
                                coordinates_list.append([])

        logger.info(f"[OCR OK] Procesado: {len(text_lines)} bloques detectados")

    except Exception as e:
        logger.error(f"[OCR ERROR] Error procesando resultado OCR: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return text_lines, confidences, coordinates_list


def compose_pdf_ocr(base_source, ocr_data, output_spdf, is_scanned):
    """
    Crear PDF de una pagina con OCR superpuesto

    Args:
        base_source: Path a imagen PNG (escaneada) o PDF original (vectorial)
        ocr_data: Tupla (text_lines, confidences, coordinates) del OCR
        output_spdf: Path donde guardar el PDF resultante
        is_scanned: True para paginas escaneadas, False para vectoriales
    """
    try:
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from PIL import Image

        text_lines, confidences, coordinates = ocr_data

        logger.info(f"[COMPOSE_PDF] Creando PDF {'escaneado' if is_scanned else 'vectorial'} - Base: {base_source}")
        logger.info(f"[COMPOSE_PDF] OCR: {len(text_lines)} bloques de texto")

        # DETECCION UNIFICADA DE DPI
        # Para ambos flujos necesitamos saber el DPI del PNG procesado
        if is_scanned:
            # El PNG es directamente base_source
            png_path = base_source
        else:
            # Para vectorial, el PNG esta en una ruta relacionada
            png_path = base_source.replace('.pdf', '.png').replace('_2.1.page-', '_2.4.page-')

        # Detectar DPI del PNG
        try:
            img_for_dpi = Image.open(png_path)
            source_dpi = img_for_dpi.info.get('dpi', (288, 288))
            if isinstance(source_dpi, tuple):
                source_dpi = source_dpi[0]  # Usar DPI X
            logger.info(f"[COMPOSE_PDF] DPI detectado del PNG: {source_dpi}")
        except Exception as e:
            source_dpi = 288  # Valor por defecto
            logger.warning(f"[COMPOSE_PDF] No se pudo detectar DPI ({e}), asumiendo {source_dpi}")

        if is_scanned:
            # FLUJO ESCANEADO: Imagen como base + texto OCR superpuesto

            # Cargar imagen PNG
            image = Image.open(base_source)
            img_width, img_height = image.size

            # Convertir pixeles a puntos PDF usando el DPI detectado
            pdf_width = (img_width * 72) / source_dpi
            pdf_height = (img_height * 72) / source_dpi

            # Crear PDF con imagen de fondo
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=(pdf_width, pdf_height))

            # Convertir imagen a formato compatible para ReportLab
            img_buffer = io.BytesIO()
            if image.mode in ['RGBA', 'P']:
                image = image.convert('RGB')
            image.save(img_buffer, format='JPEG', quality=75, optimize=True)
            img_buffer.seek(0)

            # Dibujar imagen de fondo
            c.drawImage(ImageReader(img_buffer), 0, 0, pdf_width, pdf_height)

            # Superponer texto OCR invisible
            for i, text in enumerate(text_lines):
                if i < len(coordinates) and len(coordinates[i]) > 0:
                    confidence = confidences[i] if i < len(confidences) else 0.0

                    # Filtrar texto con baja confianza
                    if confidence < 0.3:
                        continue

                    try:
                        coords = coordinates[i]

                        # Calcular coordenadas del texto
                        x_coords = [point[0] for point in coords]
                        y_coords = [point[1] for point in coords]

                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # Convertir coordenadas de imagen a PDF
                        x_pdf = (x_min / img_width) * pdf_width
                        y_pdf = pdf_height - (y_max / img_height) * pdf_height
                        height_pdf = ((y_max - y_min) / img_height) * pdf_height

                        # Calcular tamano de fuente
                        font_size = max(6, min(height_pdf * 0.8, 20))

                        # Dibujar texto invisible para busqueda
                        c.setFillColorRGB(1, 1, 1, alpha=0.01)  # Casi transparente
                        c.setFont("Helvetica", font_size)
                        c.drawString(x_pdf, y_pdf, text)

                    except Exception as e:
                        logger.debug(f"[COMPOSE_PDF] Error posicionando texto '{text}': {e}")
                        continue

            c.save()

        else:
            # FLUJO VECTORIAL: PDF original como base + texto OCR de imagenes

            import PyPDF2
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            # Leer PDF original
            pdf_file = open(base_source, 'rb')
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            original_page = pdf_reader.pages[0]

            # Obtener dimensiones de la pagina original
            media_box = original_page.mediabox
            page_width = float(media_box.width)
            page_height = float(media_box.height)

            # Calcular factor de escala: del DPI del PNG a 72 DPI del PDF
            scale_factor = 72.0 / source_dpi
            logger.info(f"[COMPOSE_PDF] Factor de escala: {scale_factor:.3f} ({source_dpi} DPI -> 72 DPI)")

            # Crear PDF temporal con solo texto OCR
            ocr_buffer = io.BytesIO()
            c = canvas.Canvas(ocr_buffer, pagesize=(page_width, page_height))

            # Solo superponer texto OCR (de imagenes extraidas)
            for i, text in enumerate(text_lines):
                if i < len(coordinates) and len(coordinates[i]) > 0:
                    confidence = confidences[i] if i < len(confidences) else 0.0

                    if confidence < 0.3:
                        continue

                    try:
                        coords = coordinates[i]

                        # Las coordenadas del OCR vienen del PNG a source_dpi
                        # Necesitamos escalarlas a 72 DPI para el PDF
                        x_coords = [point[0] for point in coords]
                        y_coords = [point[1] for point in coords]

                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # APLICAR FACTOR DE ESCALA a las coordenadas
                        x_pdf = x_min * scale_factor
                        y_pdf = page_height - (y_max * scale_factor)
                        height_pdf = (y_max - y_min) * scale_factor

                        font_size = max(6, min(height_pdf * 0.8, 20))

                        logger.debug(f"[COMPOSE_PDF] Texto '{text[:20]}...': orig({x_min:.0f},{y_min:.0f}) -> pdf({x_pdf:.0f},{y_pdf:.0f})")

                        # Dibujar texto invisible
                        c.setFillColorRGB(1, 1, 1, alpha=0.01)
                        c.setFont("Helvetica", font_size)
                        c.drawString(x_pdf, y_pdf, text)

                    except Exception as e:
                        logger.debug(f"[COMPOSE_PDF] Error posicionando texto vectorial '{text}': {e}")
                        continue

            # Asegurar que siempre hay una pagina aunque no haya texto
            if len(text_lines) == 0:
                c.showPage()  # Crear pagina vacia

            c.save()
            ocr_buffer.seek(0)

            # Combinar PDF original con capa OCR
            from PyPDF2 import PdfWriter

            pdf_writer = PdfWriter()

            # Leer capa OCR
            ocr_pdf = PyPDF2.PdfReader(ocr_buffer)
            ocr_page = ocr_pdf.pages[0]

            # Superponer OCR sobre pagina original
            original_page.merge_page(ocr_page)
            pdf_writer.add_page(original_page)

            # Cerrar archivo
            pdf_file.close()

            # Guardar resultado
            buffer = io.BytesIO()
            pdf_writer.write(buffer)

        # Guardar PDF final
        buffer.seek(0)
        with open(output_spdf, 'wb') as f:
            f.write(buffer.getvalue())

    except Exception as e:
        logger.error(f"[COMPOSE_PDF ERROR] Error creando PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Fallback: copiar archivo base
        try:
            if is_scanned:
                # Crear PDF simple con la imagen
                from PIL import Image
                image = Image.open(base_source)
                image.save(output_spdf, "PDF", resolution=288.0)
                logger.info(f"[COMPOSE_PDF] Fallback: PDF simple creado desde imagen")
            else:
                # Copiar PDF original
                subprocess.run(['cp', base_source, output_spdf], check=True)
                logger.info(f"[COMPOSE_PDF] Fallback: PDF original copiado")
        except Exception as fallback_error:
            logger.error(f"[COMPOSE_PDF] Error en fallback: {fallback_error}")
            raise


@app.route('/health')
def health():
    """Health check - responde inmediatamente para evitar reinicios"""
    global models_loaded, models_loading, models_error, startup_time

    uptime = time.time() - startup_time
    memory = get_memory_usage()

    # Siempre responde healthy para que EasyPanel no reinicie
    # El estado real se muestra en los campos adicionales
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': round(uptime, 1),
        'memory_usage': memory,
        'models_loaded': models_loaded,
        'models_loading': models_loading,
        'models_error': models_error,
        'preprocessor_ready': doc_preprocessor is not None if models_loaded else False,
        'ocr_ready': ocr_initialized if models_loaded else False,
        'opencv_config': OPENCV_CONFIG,
        'rotation_config': ROTATION_CONFIG
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    """Endpoint OCR - procesa documento completo con orientacion y OCR"""
    global doc_preprocessor, ocr_instance, ocr_initialized, ROTATION_CONFIG
    start_time = time.time()

    try:
        # 1. VALIDACION Y SETUP
        filename_param = request.form.get('filename')
        if not filename_param:
            return jsonify({'error': 'filename required'}), 400

        # Extraer paths y configuracion
        if filename_param.startswith('/'):
            full_path = filename_param
            filename = Path(full_path).name
            n8nHomeDir = str(Path(full_path).parent.parent)
        else:
            filename = filename_param
            n8nHomeDir = request.form.get('n8nHomeDir', '/home/n8n')

        base_name = Path(filename).stem
        ext = Path(filename).suffix.lower()

        logger.info("")
        logger.info("[OCR] ==========================================================================================")
        logger.info(f"[OCR] Procesando: {n8nHomeDir}/in/{filename}")
        logger.info("[OCR] ==========================================================================================")

        # Actualizar MIN_SKEW_ANGLE si se pasa como parametro
        min_angle_param = request.form.get('min_angle')
        if min_angle_param:
            try:
                ROTATION_CONFIG['MIN_SKEW_ANGLE'] = float(min_angle_param)
                logger.info(f"[OCR] MIN_SKEW_ANGLE actualizado a: {ROTATION_CONFIG['MIN_SKEW_ANGLE']}")
            except ValueError:
                logger.warning(f"[OCR] Valor invalido para min_angle: {min_angle_param}")
        else:
            ROTATION_CONFIG['MIN_SKEW_ANGLE'] = float(os.getenv('ROTATION_MIN_SKEW_ANGLE', '0.2'))

        # VERIFICAR Y CARGAR MODELOS SI ES NECESARIO
        if not doc_preprocessor:
            logger.info("[OCR] Modelo de orientacion no cargado, inicializando...")
            if not init_docpreprocessor():
                logger.warning("[OCR] No se pudo cargar modelo de orientacion")

        if not ocr_instance:
            logger.info("[OCR] Modelo OCR no cargado, inicializando...")
            if not init_ocr():
                return jsonify({'error': 'OCR initialization failed'}), 503

        # Verificar que realmente funcionan los modelos
        try:
            # Test rápido para verificar que OCR responde
            test_result = ocr_instance.predict.__name__
        except:
            logger.warning("[OCR] OCR instance no responde, reinicializando...")
            ocr_instance = None
            if not init_ocr():
                return jsonify({'error': 'OCR reinitialization failed'}), 503

        # Crear directorios necesarios
        os.makedirs(f"{n8nHomeDir}/ocr", exist_ok=True)
        os.makedirs(f"{n8nHomeDir}/pdf", exist_ok=True)

        # Verificar que existe archivo de entrada
        in_file = f"{n8nHomeDir}/in/{filename}"
        if not os.path.exists(in_file):
            return jsonify({'error': f'File not found: {in_file}'}), 404

        # PREPARAR ARCHIVO RECIBIDO
        if ext == '.pdf':
            # PREPARACION PDF
            if not init_pdf_prep(n8nHomeDir, base_name, ext):
                return jsonify({'error': 'PDF preparation failed'}), 500
        else:
            # PREPARACION IMAGEN
            if not init_img_prep(n8nHomeDir, base_name, ext):
                return jsonify({'error': 'Image preparation failed'}), 500

        # 3. PROCESAMIENTO OCR (orientacion + OCR integrado)
        logger.info("[OCR] Ejecutando procesamiento OCR completo...")
        success, message, ocr_data = proc_pdf_ocr(n8nHomeDir, base_name, ext)

        if not success:
            logger.error(f"[OCR] Error en procesamiento: {message}")
            return jsonify({'error': message}), 500

        # 4. PREPARAR RESPUESTA
        end_time = time.time()
        duration = end_time - start_time

        # Extraer datos del OCR
        text_lines = ocr_data.get('text_lines', [])
        confidences = ocr_data.get('confidences', [])
        total_blocks = ocr_data.get('total_blocks', 0)
        pages = ocr_data.get('pages', 1)

        # Calcular estadisticas
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Unir texto extraido
        full_text = '\n'.join(text_lines)

        logger.info("[OCR] ==========================================================================================")
        logger.info(f"[OCR STATS] Documento procesado correctamente - Paginas: {pages} - Tiempo {duration:.2f}s")
        logger.info("[OCR] ==========================================================================================")

        return jsonify({
            'success': True,
            'in_file': filename,
            'pdf_file': f"{base_name}.pdf",
            'extracted_text': full_text,
            'stats': {
                'total_pages': pages,
                'total_blocks': total_blocks,
                'avg_confidence': round(avg_confidence, 3),
                'processing_time': round(duration, 2)
            }
        })

    except Exception as e:
        logger.error(f"[OCR ERROR] Error en endpoint OCR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ============================================================================
# CAPA API REST AÑADIDA (WebComunica API Layer)
# Añadido para proporcionar API REST profesional sobre proyecto de Paco
# ============================================================================

# Estadísticas del servidor (para nuevos endpoints)
server_stats = {
    'startup_time': time.time(),
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_processing_time': 0.0
}

@app.route('/')
def dashboard():
    """Dashboard web interactivo"""
    uptime = int(time.time() - server_stats['startup_time'])
    success_rate = (server_stats['successful_requests'] / server_stats['total_requests'] * 100) if server_stats['total_requests'] > 0 else 0
    avg_time = (server_stats['total_processing_time'] / server_stats['successful_requests']) if server_stats['successful_requests'] > 0 else 0

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PaddleOCR Fusion v3 - Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="30">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
            border-left: 4px solid #764ba2;
            padding-left: 15px;
        }}
        .status-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .endpoint-list {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .endpoint {{
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-left: 4px solid #764ba2;
            border-radius: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 10px;
        }}
        .badge-get {{ background: #28a745; color: white; }}
        .badge-post {{ background: #007bff; color: white; }}
        .badge-new {{ background: #ffc107; color: #333; }}
        .badge-original {{ background: #6c757d; color: white; }}
        .upload-form {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border: 2px dashed #667eea;
        }}
        .upload-form input[type="file"] {{
            display: block;
            margin: 15px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: 100%;
            box-sizing: border-box;
        }}
        .upload-form button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
        }}
        .upload-form button:hover {{
            opacity: 0.9;
        }}
        .upload-form button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        #result-box {{
            background: #1a1a2e;
            color: #0f0;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            max-height: 500px;
            overflow-y: auto;
        }}
        #loading {{
            display: none;
            text-align: center;
            padding: 20px;
        }}
        .spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .feature-badge {{
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        .badge-success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 PaddleOCR Fusion v3 - Dashboard</h1>
        <p><strong>Enfoque:</strong> API REST sobre proyecto de Paco (PaddleOCR 3.x + Preprocesamiento completo)</p>

        <div class="status-box">
            <h3 style="margin-top:0;">📊 Estado del Servidor</h3>
            <p><strong>Estado:</strong> {'✅ Operativo' if (doc_preprocessor and ocr_initialized) else '⏳ Inicializando'}</p>
            <p><strong>Preprocesador:</strong> {'✅ Listo' if doc_preprocessor else '❌ No disponible'}</p>
            <p><strong>OCR:</strong> {'✅ Listo' if ocr_initialized else '❌ No disponible'}</p>
            <p><strong>Uptime:</strong> {uptime//3600}h {(uptime%3600)//60}m {uptime%60}s</p>
        </div>

        <h2>📈 Estadísticas</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{server_stats['total_requests']}</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{server_stats['successful_requests']}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_time:.2f}s</div>
                <div class="stat-label">Avg Time</div>
            </div>
        </div>

        <h2>📤 Probar OCR</h2>
        <div class="upload-form">
            <form id="ocr-form" enctype="multipart/form-data">
                <label><strong>Selecciona un archivo PDF o imagen:</strong></label>
                <input type="file" id="file-input" name="file" accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" required>
                <button type="submit" id="submit-btn">🚀 Procesar OCR</button>
            </form>
            <div id="loading">
                <div class="spinner"></div>
                <p><strong>Procesando...</strong></p>
                <p style="color:#666;">La primera vez puede tardar ~2 minutos mientras se cargan los modelos.</p>
            </div>
            <div id="result-box"></div>
        </div>

        <script>
        document.getElementById('ocr-form').addEventListener('submit', async function(e) {{
            e.preventDefault();

            const fileInput = document.getElementById('file-input');
            const submitBtn = document.getElementById('submit-btn');
            const loading = document.getElementById('loading');
            const resultBox = document.getElementById('result-box');

            if (!fileInput.files[0]) {{
                alert('Por favor selecciona un archivo');
                return;
            }}

            // Mostrar loading
            submitBtn.disabled = true;
            loading.style.display = 'block';
            resultBox.style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {{
                const response = await fetch('/process', {{
                    method: 'POST',
                    body: formData
                }});

                const result = await response.json();

                // Mostrar resultado
                resultBox.style.display = 'block';
                if (result.success) {{
                    resultBox.style.color = '#0f0';
                    resultBox.innerHTML = '<strong>✅ OCR Completado</strong>\\n\\n' +
                        '<strong>Texto extraído:</strong>\\n' + (result.text || result.full_text || 'Sin texto') +
                        '\\n\\n<strong>Tiempo:</strong> ' + (result.processing_time || 'N/A') + 's' +
                        '\\n<strong>Páginas:</strong> ' + (result.pages_processed || 'N/A');
                }} else {{
                    resultBox.style.color = '#f00';
                    resultBox.innerHTML = '<strong>❌ Error:</strong>\\n' + (result.error || 'Error desconocido');
                }}
            }} catch (error) {{
                resultBox.style.display = 'block';
                resultBox.style.color = '#f00';
                resultBox.innerHTML = '<strong>❌ Error de conexión:</strong>\\n' + error.message;
            }} finally {{
                submitBtn.disabled = false;
                loading.style.display = 'none';
            }}
        }});
        </script>

        <h2>🎯 Características del Proyecto</h2>
        <div>
            <span class="feature-badge badge-success">✅ PaddleOCR 3.x (Base de Paco)</span>
            <span class="feature-badge badge-success">✅ Preprocesamiento OpenCV</span>
            <span class="feature-badge badge-success">✅ Corrección de Perspectiva</span>
            <span class="feature-badge badge-success">✅ Corrección de Orientación</span>
            <span class="feature-badge badge-success">✅ Corrección de Inclinación</span>
            <span class="feature-badge badge-success">✅ Multi-página Inteligente</span>
            <span class="feature-badge badge-success">✅ Integración n8n</span>
            <span class="feature-badge badge-success">✅ API REST (Nueva)</span>
            <span class="feature-badge badge-success">✅ Dashboard Web (Nuevo)</span>
        </div>

        <h2>🔌 Endpoints API</h2>
        <div class="endpoint-list">
            <div class="endpoint">
                <span class="badge badge-get">GET</span>
                <span class="badge badge-new">NUEVO</span>
                <code>/</code> - Dashboard web interactivo
            </div>
            <div class="endpoint">
                <span class="badge badge-get">GET</span>
                <span class="badge badge-new">NUEVO</span>
                <code>/health</code> - Health check completo
            </div>
            <div class="endpoint">
                <span class="badge badge-get">GET</span>
                <span class="badge badge-new">NUEVO</span>
                <code>/stats</code> - Estadísticas detalladas del servidor
            </div>
            <div class="endpoint">
                <span class="badge badge-post">POST</span>
                <span class="badge badge-new">NUEVO</span>
                <code>/process</code> - OCR estándar (wrapper REST sobre /ocr)
            </div>
            <div class="endpoint">
                <span class="badge badge-post">POST</span>
                <span class="badge badge-new">NUEVO</span>
                <code>/analyze</code> - Análisis ultra-detallado con visualización
            </div>
            <div class="endpoint">
                <span class="badge badge-post">POST</span>
                <span class="badge badge-original">ORIGINAL</span>
                <code>/ocr</code> - Endpoint original de Paco (integración n8n)
            </div>
        </div>

        <h2>📖 Ejemplos de Uso</h2>

        <h3>API REST estándar (nuevos endpoints):</h3>
        <pre style="background:#f4f4f4; padding:15px; border-radius:5px; overflow-x:auto;">
# OCR estándar
curl -X POST http://localhost:8503/process \\
  -F "file=@documento.pdf" \\
  -F "language=es" \\
  -F "detailed=true"

# Análisis detallado
curl -X POST http://localhost:8503/analyze \\
  -F "file=@documento.pdf" | jq -r '.ultra_analysis'

# Estadísticas
curl http://localhost:8503/stats | jq
        </pre>

        <h3>Integración n8n (endpoint original):</h3>
        <pre style="background:#f4f4f4; padding:15px; border-radius:5px; overflow-x:auto;">
# Usar endpoint /ocr (mantiene compatibilidad total)
curl -X POST http://localhost:8503/ocr \\
  -F "filename=/home/n8n/in/documento.pdf"
        </pre>

        <h2>📚 Documentación</h2>
        <ul>
            <li><a href="https://github.com/PaddlePaddle/PaddleOCR">PaddleOCR Official Docs</a></li>
            <li><strong>README.md</strong> - Documentación completa del proyecto</li>
            <li><strong>CLAUDE.md</strong> - Guía para desarrollo</li>
        </ul>

        <p style="text-align:center; color:#999; margin-top:30px;">
            <em>Dashboard actualizado automáticamente cada 30 segundos</em><br>
            <strong>PaddleOCR Fusion v3.0.0</strong> - API REST sobre proyecto de Paco
        </p>
    </div>
</body>
</html>
    """
    return html


@app.route('/stats')
def stats():
    """Estadísticas detalladas del servidor"""
    uptime = int(time.time() - server_stats['startup_time'])
    success_rate = (server_stats['successful_requests'] / server_stats['total_requests'] * 100) if server_stats['total_requests'] > 0 else 0
    avg_time = (server_stats['total_processing_time'] / server_stats['successful_requests']) if server_stats['successful_requests'] > 0 else 0

    return jsonify({
        'status': 'healthy' if (doc_preprocessor and ocr_initialized) else 'initializing',
        'uptime_seconds': uptime,
        'uptime_formatted': f"{uptime//3600}h {(uptime%3600)//60}m {uptime%60}s",
        'preprocessor_ready': doc_preprocessor is not None,
        'ocr_ready': ocr_initialized,
        'statistics': {
            'total_requests': server_stats['total_requests'],
            'successful_requests': server_stats['successful_requests'],
            'failed_requests': server_stats['failed_requests'],
            'success_rate': round(success_rate, 2),
            'total_processing_time': round(server_stats['total_processing_time'], 2),
            'avg_processing_time': round(avg_time, 3)
        },
        'configuration': {
            'opencv_config': OPENCV_CONFIG,
            'rotation_config': ROTATION_CONFIG
        },
        'version': '3.0.0-fusion',
        'base_project': 'paddlepaddle_paco',
        'api_layer': 'webcomunica'
    })


@app.route('/process', methods=['POST'])
def process():
    """
    Endpoint REST estándar - Wrapper sobre el endpoint /ocr de Paco
    Acepta archivos multipart en lugar de rutas en disco
    """
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1

    temp_file_path = None

    try:
        # Validar archivo
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'Empty filename'}), 400

        # Validar extensión
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            server_stats['failed_requests'] += 1
            return jsonify({'error': f'Unsupported file format: {ext}'}), 400

        # Obtener parámetros
        language = request.form.get('language', 'es')
        detailed = request.form.get('detailed', 'false').lower() == 'true'

        # Guardar archivo temporal en /home/n8n/in para compatibilidad con /ocr
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)
        os.makedirs(f"{n8nHomeDir}/ocr", exist_ok=True)
        os.makedirs(f"{n8nHomeDir}/pdf", exist_ok=True)

        temp_filename = f"temp_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        logger.info(f"[PROCESS] Archivo guardado temporalmente: {temp_file_path}")

        # Llamar al endpoint /ocr internamente usando la lógica de Paco
        # Creamos un request simulado
        with app.test_request_context(
            '/ocr',
            method='POST',
            data={'filename': temp_file_path}
        ):
            response = ocr()

            # Extraer datos de la respuesta
            if isinstance(response, tuple):
                response_data, status_code = response
            else:
                response_data = response
                status_code = 200

            response_json = response_data.get_json()

        # Limpiar archivo temporal
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            # Limpiar archivos intermedios generados
            base_name = Path(temp_filename).stem
            for pattern in [f"{n8nHomeDir}/ocr/{base_name}*", f"{n8nHomeDir}/pdf/{base_name}*"]:
                import glob
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"[PROCESS] Error limpiando archivos temporales: {e}")

        # Preparar respuesta en formato REST estándar
        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        if response_json.get('success'):
            server_stats['successful_requests'] += 1

            result = {
                'success': True,
                'text': response_json.get('extracted_text', ''),
                'stats': response_json.get('stats', {}),
                'processing_time': round(processing_time, 3),
                'timestamp': time.time()
            }

            if detailed:
                result['detailed_stats'] = {
                    'total_pages': response_json['stats'].get('total_pages', 1),
                    'total_blocks': response_json['stats'].get('total_blocks', 0),
                    'avg_confidence': response_json['stats'].get('avg_confidence', 0.0)
                }

            return jsonify(result)
        else:
            server_stats['failed_requests'] += 1
            return jsonify({
                'success': False,
                'error': response_json.get('error', 'Unknown error'),
                'processing_time': round(processing_time, 3)
            }), 500

    except Exception as e:
        server_stats['failed_requests'] += 1
        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        logger.error(f"[PROCESS ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Limpiar archivo temporal en caso de error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time': round(processing_time, 3)
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Análisis ultra-detallado con visualización
    Similar a /process pero con salida formateada para debugging
    """
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1

    temp_file_path = None

    try:
        # Validar archivo
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'Empty filename'}), 400

        # Guardar archivo temporal
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)

        temp_filename = f"analyze_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        # Llamar al endpoint /ocr
        with app.test_request_context(
            '/ocr',
            method='POST',
            data={'filename': temp_file_path}
        ):
            response = ocr()
            if isinstance(response, tuple):
                response_data, status_code = response
            else:
                response_data = response
                status_code = 200

            response_json = response_data.get_json()

        # Limpiar archivos temporales
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            base_name = Path(temp_filename).stem
            import glob
            for pattern in [f"{n8nHomeDir}/ocr/{base_name}*", f"{n8nHomeDir}/pdf/{base_name}*"]:
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"[ANALYZE] Error limpiando archivos: {e}")

        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        if response_json.get('success'):
            server_stats['successful_requests'] += 1

            # Formatear texto para visualización
            text = response_json.get('extracted_text', '')
            stats = response_json.get('stats', {})

            ultra_analysis = f"""
╔══════════════════════════════════════════════════════════════╗
║              ANÁLISIS ULTRA-DETALLADO - FUSION v3            ║
╚══════════════════════════════════════════════════════════════╝

📊 ESTADÍSTICAS:
   • Total de Páginas: {stats.get('total_pages', 1)}
   • Total de Bloques: {stats.get('total_blocks', 0)}
   • Confianza Promedio: {stats.get('avg_confidence', 0)*100:.1f}%
   • Tiempo de Procesamiento: {processing_time:.2f}s

📄 TEXTO EXTRAÍDO:
{text}

════════════════════════════════════════════════════════════════
"""

            return jsonify({
                'success': True,
                'ultra_analysis': ultra_analysis,
                'stats': stats,
                'processing_time': round(processing_time, 3),
                'configuration': 'PaddleOCR 3.x + Preprocesamiento Completo (Paco)'
            })
        else:
            server_stats['failed_requests'] += 1
            return jsonify({
                'success': False,
                'error': response_json.get('error', 'Unknown error')
            }), 500

    except Exception as e:
        server_stats['failed_requests'] += 1
        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        logger.error(f"[ANALYZE ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())

        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time': round(processing_time, 3)
        }), 500


# ============================================================================
# FIN DE CAPA API REST AÑADIDA
# ============================================================================

def start_model_loading():
    """Inicia la carga de modelos en segundo plano"""
    logger.info("[STARTUP] Iniciando hilo de carga de modelos...")
    model_thread = threading.Thread(target=load_models_background, daemon=True)
    model_thread.start()
    logger.info("[STARTUP] Hilo de carga de modelos iniciado")

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', '8503'))
    logger.info("")
    logger.info("=" * 60)
    logger.info("[START] PADDLEOCR V3 FUSION - WEBCOMUNICA")
    logger.info("=" * 60)
    logger.info(f"[START] Puerto: {port}")
    logger.info(f"[START] Memoria: {get_memory_usage()}")
    logger.info(f"[START] PID: {os.getpid()}")
    logger.info("[START] Proyecto base: paddlepaddle_paco")
    logger.info("[START] Capa API: webcomunica REST layer")

    # NO cargar modelos automáticamente - se cargarán con la primera petición OCR
    # Esto permite que el contenedor arranque rápido y sobreviva en EasyPanel
    logger.info("[STARTUP] Modelos se cargarán con la primera petición OCR (on-demand)")

    # Detectar si estamos en produccion
    if os.getenv('FLASK_ENV') == 'production':
        from waitress import serve
        logger.info("")
        logger.info("[READY] *** SERVIDOR WAITRESS INICIANDO ***")
        logger.info(f"[READY] URL: http://0.0.0.0:{port}")
        logger.info("[READY] Health check: /health")
        logger.info("[READY] Dashboard: /")
        logger.info("[READY] Modelos: se cargan on-demand (primera petición)")
        logger.info("")
        # Waitress bloqueará aquí - el servidor estará corriendo
        serve(app, host='0.0.0.0', port=port, threads=4)
    else:
        logger.info("[READY] Iniciando servidor Flask (desarrollo)")
        app.run(host='0.0.0.0', port=port, debug=False)

    # Este código solo se ejecuta si el servidor termina
    logger.info("[SHUTDOWN] Servidor terminado")

