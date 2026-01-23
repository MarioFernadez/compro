# processor.py
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
from PIL import Image
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
# UTILIDADES
# =====================================================

def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalize_currency(text: str) -> Optional[str]:
    t = text.upper()
    if any(x in t for x in ["PYG", "GUARANI", "GUARANÍ", "GS", "₲"]):
        return "PYG"
    if any(x in t for x in ["ARS", "ARG", "PESO", "$"]):
        return "ARS"
    return None


def _parse_amount(text: str) -> Optional[float]:
    candidates = re.findall(
        r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\$\s?\d+(?:[.,]\d{2}))",
        text,
    )

    def to_float(s: str) -> Optional[float]:
        try:
            s = s.replace("$", "").strip()
            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
            elif "," in s:
                s = s.replace(",", ".")
            return float(s)
        except Exception:
            return None

    values = [to_float(c) for c in candidates]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _parse_date(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{2})\s+de\s+(\w+)\s+de\s+(\d{4})", text, re.IGNORECASE)
    if m:
        months = {
            "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
            "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
            "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
        }
        return f"{m.group(3)}-{months.get(m.group(2).lower(), '01')}-{m.group(1).zfill(2)}"

    return None


def _parse_operation_id(text: str) -> Optional[str]:
    m = re.search(r"(N[ÚU]MERO\s+DE\s+OPERACI[ÓO]N.*?)(\d{8,})", text, re.IGNORECASE)
    if m:
        return m.group(2)
    return None


# =====================================================
# OCR
# =====================================================

def easyocr_extract_text(image: Image.Image) -> str:
    max_width = 1600
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)))

    text = "\n".join(
        EASYOCR_READER.readtext(np.array(image), detail=0, paragraph=True)
    )

    # 🔒 LIMPIEZA CLAVE: eliminar horarios
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    text = re.sub(r"\bhs\b", "", text, flags=re.IGNORECASE)

    return text


# =====================================================
# GEMINI VIA REST
# =====================================================

def gemini_extract_structured(ocr_text: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY no definida")
        return None

    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash:generateContent"
        )

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"""
Extraé datos de este comprobante argentino/paraguayo.
Respondé SOLO JSON válido.

Reglas ESTRICTAS:
- amount debe ser un MONTO monetario (ignorar horas, fechas, IDs)
- Priorizar montos con $
- Elegir el monto mayor
- NO inventar datos

Formato:

{{
  "emitter": string|null,
  "recipient": string|null,
  "amount": number|null,
  "currency": "ARS"|"PYG"|null,
  "date": "YYYY-MM-DD"|null,
  "operation_id": string|null
}}

TEXTO OCR:
{ocr_text}
"""
                }]
            }]
        }

        r = requests.post(
            f"{url}?key={api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if r.status_code != 200:
            print("❌ Gemini HTTP:", r.text)
            return None

        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        print("✅ Gemini ACTIVO")
        return json.loads(text)

    except Exception as e:
        print("❌ Gemini ERROR:", e)
        return None


# =====================================================
# PIPELINE PRINCIPAL
# =====================================================

def extract_all(image_bytes: bytes) -> ExtractedData:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    ocr_text = easyocr_extract_text(image)
    data = ExtractedData(raw_text=ocr_text)

    structured = gemini_extract_structured(ocr_text)

    if structured:
        data.emitter = structured.get("emitter")
        data.recipient = structured.get("recipient")
        data.amount = structured.get("amount")
        data.currency = structured.get("currency")
        data.date = structured.get("date")
        data.operation_id = structured.get("operation_id")
        data.extracted_json = json.dumps(structured, ensure_ascii=False)
        return data

    # fallback
    data.currency = _normalize_currency(ocr_text)
    data.amount = _parse_amount(ocr_text)
    data.date = _parse_date(ocr_text)
    data.operation_id = _parse_operation_id(ocr_text)

    data.extracted_json = json.dumps(
        {"source": "fallback", "raw_text": ocr_text},
        ensure_ascii=False,
    )

    return data
