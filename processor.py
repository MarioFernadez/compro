from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image, ImageOps
import easyocr
import requests


# =====================================================
# EASYOCR: inicializar UNA sola vez
# =====================================================
# Tip: al setear EASYOCR_MODULE_PATH en Dockerfile, los modelos quedan cacheados en /app/.easyocr
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
# CACHE SIMPLE EN MEMORIA (evita reprocesos si Streamlit rerun)
# =====================================================
_OCR_CACHE: Dict[str, str] = {}
_CACHE_MAX = 256


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _cache_get(sha: str) -> Optional[str]:
    return _OCR_CACHE.get(sha)


def _cache_set(sha: str, text: str) -> None:
    if sha in _OCR_CACHE:
        _OCR_CACHE[sha] = text
        return
    if len(_OCR_CACHE) >= _CACHE_MAX:
        # pop un item (FIFO simple)
        _OCR_CACHE.pop(next(iter(_OCR_CACHE)))
    _OCR_CACHE[sha] = text


# =====================================================
# NORMALIZADORES / PARSERS
# =====================================================
def _normalize_spaces(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _normalize_currency(text: str) -> Optional[str]:
    t = text.upper()
    if any(x in t for x in ["PYG", "GUARANI", "GUARANÍ", "GS", "₲"]):
        return "PYG"
    # En comprobantes AR, casi siempre aparece "$"
    if any(x in t for x in ["ARS", "ARG", "PESO", "$"]):
        return "ARS"
    return None


def _parse_amount(text: str) -> Optional[float]:
    """
    Toma el monto más grande.
    Soporta:
      $ 124.740
      $124,740.50
      42.000
      42000
    """
    t = text.replace("\u202f", " ").replace("\xa0", " ")

    # candidatos con $ o números grandes con separadores
    candidates = re.findall(
        r"(?:\$\s*)?(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|\d{4,}(?:[.,]\d{2})?)",
        t,
        flags=re.IGNORECASE,
    )

    def to_float(s: str) -> Optional[float]:
        try:
            s = s.strip()
            # si tiene ambos . y , decidir separador decimal por la última aparición
            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    # 12.345,67 -> 12345.67
                    s = s.replace(".", "").replace(",", ".")
                else:
                    # 12,345.67 -> 12345.67
                    s = s.replace(",", "")
            elif "," in s:
                # puede ser decimal o miles; si hay 1 coma y 3 dígitos después -> miles
                if re.search(r",\d{3}\b", s):
                    s = s.replace(",", "")
                else:
                    s = s.replace(",", ".")
            elif "." in s:
                # si hay puntos de miles (1.234.567) -> sacar puntos
                if re.search(r"\.\d{3}\b", s):
                    s = s.replace(".", "")
            return float(s)
        except Exception:
            return None

    values = [to_float(c) for c in candidates]
    values = [v for v in values if v is not None]

    # filtrar valores que parecen horas/IDs (muy chicos)
    values = [v for v in values if v >= 1.0]

    return max(values) if values else None


def _parse_date(text: str) -> Optional[str]:
    """
    Soporta:
      '05 de enero de 2026'
      'Lunes, 05 de enero de 2026'
    """
    m = re.search(r"\b(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\b", text, re.IGNORECASE)
    if not m:
        return None

    months = {
        "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
        "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
        "septiembre": "09", "setiembre": "09",
        "octubre": "10", "noviembre": "11", "diciembre": "12",
    }
    dd = m.group(1).zfill(2)
    mm = months.get(m.group(2).lower(), "01")
    yyyy = m.group(3)
    return f"{yyyy}-{mm}-{dd}"


def _parse_operation_id(text: str) -> Optional[str]:
    """
    Mercado Pago suele decir:
      'Número de operación de Mercado Pago 140076...'
    """
    patterns = [
        r"N[ÚU]MERO\s+DE\s+OPERACI[ÓO]N(?:\s+DE\s+MERCADO\s+PAGO)?\s*[:\-]?\s*(\d{8,})",
        r"OPERACI[ÓO]N\s*[:\-]?\s*(\d{8,})",
        r"ID\s+OPERACI[ÓO]N\s*[:\-]?\s*(\d{8,})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _clean_name_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[•\-\–\*]+\s*", "", s)
    # cortar si vienen datos luego del nombre
    s = re.split(r"\b(CUIT|CUIL|CBU|CVU|BANCO|MERCADO\s+PAGO|N[ÚU]MERO|CODIGO|C[ÓO]DIGO|ID)\b", s, flags=re.IGNORECASE)[0]
    s = re.sub(r"\s{2,}", " ", s).strip(" -:•")
    return s.strip()


def _parse_emitter_recipient(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Busca bloques:
      De
      Nombre Apellido
    y
      Para
      Nombre Apellido
    """
    t = _normalize_spaces(text)

    # 1) patrones tipo:
    #   De
    #   Sandra Gabriela Diaz
    de_patterns = [
        r"\bDe\b\s*[:\-]?\s*\n\s*([^\n]{3,80})",
        r"\bDe\b\s*[:\-]?\s*([^\n]{3,80})",
    ]
    para_patterns = [
        r"\bPara\b\s*[:\-]?\s*\n\s*([^\n]{3,80})",
        r"\bPara\b\s*[:\-]?\s*([^\n]{3,80})",
    ]

    emitter = None
    recipient = None

    for p in de_patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            candidate = _clean_name_line(m.group(1))
            if candidate and len(candidate.split()) >= 2:
                emitter = candidate
                break

    for p in para_patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            candidate = _clean_name_line(m.group(1))
            if candidate and len(candidate.split()) >= 2:
                recipient = candidate
                break

    # 2) Fallback extra: si OCR metió todo en una sola línea con "De Sandra..." / "Para Romina..."
    if not emitter:
        m = re.search(r"\bDe\b\s+([A-ZÁÉÍÓÚÑ][^\n]{3,80})", t, re.IGNORECASE)
        if m:
            c = _clean_name_line(m.group(1))
            if c and len(c.split()) >= 2:
                emitter = c

    if not recipient:
        m = re.search(r"\bPara\b\s+([A-ZÁÉÍÓÚÑ][^\n]{3,80})", t, re.IGNORECASE)
        if m:
            c = _clean_name_line(m.group(1))
            if c and len(c.split()) >= 2:
                recipient = c

    # Evitar que devuelva "Mercado Pago" como nombre
    if emitter and re.search(r"mercado\s+pago", emitter, re.IGNORECASE):
        emitter = None
    if recipient and re.search(r"mercado\s+pago", recipient, re.IGNORECASE):
        recipient = None

    return emitter, recipient


# =====================================================
# OCR (rápido)
# =====================================================
def easyocr_extract_text(image: Image.Image) -> str:
    """
    Optimizado para CPU:
    - Reduce resolución (mucho más rápido)
    - Grayscale + autocontrast (mejora lectura)
    """
    max_width = 1100  # 🔥 antes 1600, esto acelera fuerte
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)))

    # preproceso
    img = image.convert("L")
    img = ImageOps.autocontrast(img)

    # OCR
    lines = EASYOCR_READER.readtext(np.array(img), detail=0, paragraph=True)

    text = "\n".join(lines)
    text = _normalize_spaces(text)

    # limpieza: eliminar horarios típicos
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    text = re.sub(r"\bhs\b", "", text, flags=re.IGNORECASE)
    return _normalize_spaces(text)


# =====================================================
# GEMINI VIA REST (opcional)
# =====================================================
def gemini_extract_structured(ocr_text: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
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

Reglas:
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
            timeout=10,  # ✅ menor timeout para que no "cuelgue"
        )

        if r.status_code != 200:
            return None

        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)

    except Exception:
        return None


# =====================================================
# PIPELINE PRINCIPAL
# =====================================================
def extract_all(image_bytes: bytes) -> ExtractedData:
    img_sha = sha256_bytes(image_bytes)

    cached = _cache_get(img_sha)
    if cached is not None:
        ocr_text = cached
    else:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ocr_text = easyocr_extract_text(image)
        _cache_set(img_sha, ocr_text)

    data = ExtractedData(raw_text=ocr_text)

    # 1) Primero: heurísticas rápidas (MercadoPago De/Para)
    emitter, recipient = _parse_emitter_recipient(ocr_text)
    data.emitter = emitter
    data.recipient = recipient

    # 2) Monto / moneda / fecha / operación (rápido)
    data.currency = _normalize_currency(ocr_text)
    data.amount = _parse_amount(ocr_text)
    data.date = _parse_date(ocr_text)
    data.operation_id = _parse_operation_id(ocr_text)

    # 3) Si hay Gemini y querés mejorar, lo usamos (pero sin bloquear si falla)
    structured = gemini_extract_structured(ocr_text)
    if structured:
        # Solo pisa si trae algo mejor
        data.emitter = structured.get("emitter") or data.emitter
        data.recipient = structured.get("recipient") or data.recipient
        data.amount = structured.get("amount") if structured.get("amount") is not None else data.amount
        data.currency = structured.get("currency") or data.currency
        data.date = structured.get("date") or data.date
        data.operation_id = structured.get("operation_id") or data.operation_id
        data.extracted_json = json.dumps(structured, ensure_ascii=False)
        return data

    data.extracted_json = json.dumps(
        {
            "source": "fallback_fast",
            "emitter": data.emitter,
            "recipient": data.recipient,
            "amount": data.amount,
            "currency": data.currency,
            "date": data.date,
            "operation_id": data.operation_id,
        },
        ensure_ascii=False,
    )
    return data
W