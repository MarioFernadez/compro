from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance
import easyocr
import requests


# =====================================================
# INICIALIZACIONES GLOBALES
# =====================================================

# EasyOCR se inicializa UNA SOLA VEZ
EASYOCR_READER = easyocr.Reader(["es"], gpu=False)


# =====================================================
# MODELO DE DATOS
# =====================================================

@dataclass
class ExtractedData:
    emitter: Optional[str] = None
    recipient: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    operation_id: Optional[str] = None

    raw_text: Optional[str] = None
    extracted_json: Optional[str] = None


# =====================================================
# UTILIDADES DE LIMPIEZA Y FALLBACK
# =====================================================

def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalize_currency(text: str) -> Optional[str]:
    t = text.upper()
    # Prioridad a Paraguay si detecta ueno o Gs
    if any(x in t for x in ["UENO", "GS", "₲", "GUARAN", "FAMILIAR"]):
        return "PYG"
    if any(x in t for x in ["ARS", "ARG", "PESO", "$"]):
        return "ARS"
    return None


def _parse_amount(text: str) -> Optional[float]:
    # Regex mejorada para detectar montos con puntos de miles (PY) o decimales (AR)
    candidates = re.findall(r"(?:gs\.?|gs|\$)\s?([\d\.,]{3,})", text, re.IGNORECASE)
    
    def to_float(s: str) -> Optional[float]:
        try:
            # Si tiene más de un punto, es probable que sean miles (Paraguay)
            if s.count('.') >= 1 and ',' not in s:
                s = s.replace('.', '')
            # Si tiene coma y punto, estandarizamos a punto decimal
            elif ',' in s and '.' in s:
                if s.rfind(',') > s.rfind('.'):
                    s = s.replace('.', '').replace(',', '.')
                else:
                    s = s.replace(',', '')
            elif ',' in s:
                s = s.replace(',', '.')
            return float(s)
        except:
            return None

    values = [to_float(c) for c in candidates]
    valid_values = [v for v in values if v is not None and v > 0]
    return max(valid_values) if valid_values else None


# =====================================================
# OCR CON PREPROCESAMIENTO
# =====================================================

def easyocr_extract_text(image: Image.Image) -> str:
    # --- MEJORA DE IMAGEN PARA FOTOS DE PANTALLA ---
    # Aumentar contraste y nitidez antes del OCR
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)
    
    max_width = 1600
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)))

    results = EASYOCR_READER.readtext(np.array(image), detail=0, paragraph=False)
    text = "\n".join(results)

    # Limpieza de basura común en OCR
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text) # Quitar horas
    return text


# =====================================================
# GEMINI CON PROMPT UNIVERSAL (PY/AR)
# =====================================================

def gemini_extract_structured(ocr_text: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

        prompt = f"""
Actúa como un experto en contabilidad de Argentina y Paraguay. 
Analiza el texto de este comprobante de transferencia y extrae los datos exactos.

REGLAS DE ORO:
1. 'amount': Busca el monto principal. En Paraguay (Gs.) suele no tener decimales. En Argentina ($) suele tener dos decimales.
2. 'currency': Si ves "Gs", "Guaranies", "ueno" o "Familiar", es "PYG". Si ves "$" o "Mercado Pago" o "Galicia", es "ARS".
3. 'recipient': Nombre de la persona o empresa que recibe el dinero.
4. 'emitter': Nombre de quien envía el dinero.
5. 'date': Convertir a formato YYYY-MM-DD.

TEXTO DEL COMPROBANTE:
{ocr_text}

Responde exclusivamente con un objeto JSON:
{{
  "emitter": string|null,
  "recipient": string|null,
  "amount": number|null,
  "currency": "ARS"|"PYG",
  "date": "YYYY-MM-DD"|null,
  "operation_id": string|null
}}
"""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        r = requests.post(
            f"{url}?key={api_key}",
            json=payload,
            timeout=30
        )

        if r.status_code == 200:
            res_json = r.json()
            content = res_json["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(content)
        return None

    except Exception as e:
        print(f"Error Gemini: {e}")
        return None


# =====================================================
# PIPELINE PRINCIPAL
# =====================================================

def extract_all(image_bytes: bytes) -> ExtractedData:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1. OCR
    ocr_text = easyocr_extract_text(image)
    data = ExtractedData(raw_text=ocr_text)

    # 2. Intento con Gemini (Cerebro principal)
    structured = gemini_extract_structured(ocr_text)

    if structured and structured.get("amount"):
        data.emitter = structured.get("emitter")
        data.recipient = structured.get("recipient")
        data.amount = float(structured.get("amount"))
        data.currency = structured.get("currency")
        data.date = structured.get("date")
        data.operation_id = str(structured.get("operation_id")) if structured.get("operation_id") else None
        data.extracted_json = json.dumps(structured, ensure_ascii=False)
        return data

    # 3. Fallback (Si Gemini falla o no hay internet)
    data.currency = _normalize_currency(ocr_text) or "ARS"
    data.amount = _parse_amount(ocr_text)
    
    data.extracted_json = json.dumps(
        {"source": "fallback", "amount": data.amount, "currency": data.currency},
        ensure_ascii=False,
    )

    return data