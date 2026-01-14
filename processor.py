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

# Gemini (opcional, tolerante a fallos)
try:
    import google.generativeai as genai
except Exception:
    genai = None


# =====================================================
# INICIALIZACIONES GLOBALES (CLAVE PARA NO COLGAR)
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
    currency: Optional[str] = None  # ARS / PYG
    date: Optional[str] = None      # YYYY-MM-DD
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
        r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))(?!\d)",
        text,
    )

    def to_float(s: str) -> Optional[float]:
        try:
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
    m = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.search(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b", text)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    return None


def _parse_operation_id(text: str) -> Optional[str]:
    patterns = [
        r"(?:ID\s*OPERACI[ÓO]N|OPERACI[ÓO]N|TRANSACCI[ÓO]N|COMPROBANTE|AUTORIZACI[ÓO]N)\s*[:#]?\s*([A-Z0-9\-]{6,})",
        r"\b([A-Z0-9]{10,})\b",
    ]
    t = text.upper()
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def _guess_parties(text: str) -> tuple[Optional[str], Optional[str]]:
    t = text.upper()

    def find(label: str) -> Optional[str]:
        m = re.search(label + r"\s*[:\-]\s*(.{3,80})", t)
        if not m:
            return None
        val = re.split(r"\b(DNI|CUIT|RUC|ID|OPERACI)\b", m.group(1))[0]
        return val.title().strip()

    emitter = find("EMISOR") or find("PAGADOR") or find("ORIGEN")
    recipient = find("DESTINATARIO") or find("BENEFICIARIO") or find("RECEPTOR")

    return emitter, recipient


# =====================================================
# OCR OPTIMIZADO (NO SE CUELGA)
# =====================================================

def easyocr_extract_text(image: Image.Image) -> str:
    # Redimensionar imágenes grandes para evitar cuelgues
    max_width = 1600
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)))

    image_np = np.array(image)

    result = EASYOCR_READER.readtext(
        image_np,
        detail=0,
        paragraph=True,
    )

    if isinstance(result, list):
        return "\n".join([str(x) for x in result if x])

    return str(result)


# =====================================================
# GEMINI (OPCIONAL, BLINDADO)
# =====================================================

def gemini_extract_structured(ocr_text: str) -> Optional[dict]:
    """
    Usa Gemini SOLO si está disponible.
    Si falla por modelo, permisos o API, NO rompe el sistema.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None

    try:
        genai.configure(api_key=api_key)

        # Modelo más compatible históricamente
        model = genai.GenerativeModel("models/gemini-pro")

        prompt = f"""
Extraé datos de un comprobante (Argentina o Paraguay).
Respondé SOLO JSON válido con este formato:

{{
  "emitter": string|null,
  "recipient": string|null,
  "amount": number|null,
  "currency": "ARS"|"PYG"|null,
  "date": "YYYY-MM-DD"|null,
  "operation_id": string|null
}}

Texto OCR:
{ocr_text}
""".strip()

        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()

        return json.loads(text)

    except Exception as e:
        # Falla segura: no rompe la app
        print("⚠️ Gemini no disponible:", e)
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

    # Fallback heurístico (si no hay IA)
    data.currency = _normalize_currency(ocr_text)
    data.amount = _parse_amount(ocr_text)
    data.date = _parse_date(ocr_text)
    data.operation_id = _parse_operation_id(ocr_text)
    data.emitter, data.recipient = _guess_parties(ocr_text)

    data.extracted_json = json.dumps(
        {
            "emitter": data.emitter,
            "recipient": data.recipient,
            "amount": data.amount,
            "currency": data.currency,
            "date": data.date,
            "operation_id": data.operation_id,
            "source": "heuristic_fallback",
            "extracted_at": datetime.utcnow().isoformat(),
        },
        ensure_ascii=False,
    )

    return data
