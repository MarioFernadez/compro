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

from PIL import Image

# EasyOCR (usa torch internamente)
import easyocr

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None


@dataclass
class ExtractedData:
    emitter: Optional[str] = None
    recipient: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None  # ARS/PYG
    date: Optional[str] = None      # YYYY-MM-DD
    operation_id: Optional[str] = None

    raw_text: Optional[str] = None
    extracted_json: Optional[str] = None


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalize_currency(text: str) -> Optional[str]:
    t = text.upper()
    if "PYG" in t or "GUARANI" in t or "GS" in t or "₲" in t:
        return "PYG"
    if "ARS" in t or "ARG" in t or "PESO" in t or "$" in t:
        # Ojo: $ aparece también en otros; lo tratamos como pista débil
        return "ARS"
    return None


def _parse_amount(text: str) -> Optional[float]:
    # Busca números tipo 1.234,56 / 1234.56 / 1,234.56
    candidates = re.findall(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))(?!\d)", text)
    if not candidates:
        return None

    def to_float(s: str) -> Optional[float]:
        s = s.strip()
        # Heurística: si termina en ,dd -> decimal coma
        if re.search(r",\d{2}$", s) and "." in s:
            # 1.234,56 -> miles '.' y decimal ','
            s = s.replace(".", "").replace(",", ".")
        elif re.search(r"\.\d{2}$", s) and "," in s:
            # 1,234.56 -> miles ',' y decimal '.'
            s = s.replace(",", "")
        else:
            # 1234,56 -> decimal coma
            if "," in s and "." not in s:
                s = s.replace(",", ".")
            # 1.234 -> podría ser miles sin decimales; lo ignoramos (queremos montos con 2 decimales)
        try:
            return float(s)
        except Exception:
            return None

    floats = [to_float(c) for c in candidates]
    floats = [f for f in floats if f is not None]
    if not floats:
        return None
    # normalmente el monto total es el mayor
    return max(floats)


def _parse_date(text: str) -> Optional[str]:
    # dd/mm/yyyy o dd-mm-yyyy o yyyy-mm-dd
    m = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", text)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"
    m = re.search(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b", text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"
    return None


def _parse_operation_id(text: str) -> Optional[str]:
    # Busca etiquetas típicas: "ID operación", "Nro", "Transacción", "Operacion", "Comprobante", "Autorización"
    patterns = [
        r"(?:ID\s*OPERACI[ÓO]N|ID\s*OP|OPERACI[ÓO]N|TRANSACCI[ÓO]N|NRO\.?|N[ÚU]M\.?|COMPROBANTE|AUTORIZACI[ÓO]N)\s*[:#]?\s*([A-Z0-9\-]{6,})",
        r"\b([A-Z0-9]{10,})\b",
    ]
    for p in patterns:
        m = re.search(p, text.upper())
        if m:
            return m.group(1)
    return None


def _guess_parties(text: str) -> tuple[Optional[str], Optional[str]]:
    # Muy heurístico: busca "EMISOR"/"DESTINATARIO"/"PAGADOR"/"BENEFICIARIO"
    upper = text.upper()
    emitter = None
    recipient = None

    def find_label(label: str) -> Optional[str]:
        m = re.search(label + r"\s*[:\-]\s*(.{3,80})", upper)
        if not m:
            return None
        val = m.group(1).strip()
        val = re.sub(r"\s{2,}", " ", val)
        # corta si aparece otra etiqueta
        val = re.split(r"\b(DESTINATARIO|BENEFICIARIO|PAGADOR|RECEPTOR|CUIT|RUC|DNI|ID|OPERACI)\b", val)[0].strip()
        return val.title()[:80] if val else None

    emitter = find_label("EMISOR") or find_label("PAGADOR") or find_label("ORIGEN")
    recipient = find_label("DESTINATARIO") or find_label("BENEFICIARIO") or find_label("RECEPTOR") or find_label("DESTINO")

    return emitter, recipient


def easyocr_extract_text(image: Image.Image) -> str:
    # Reader global simple (cache por proceso)
    # Nota: para Argentina/Paraguay, español suele bastar; si querés guaraní/portugués se puede ampliar.
    reader = easyocr.Reader(["es"], gpu=False)
    result = reader.readtext(image, detail=0, paragraph=True)
    if isinstance(result, list):
        return "\n".join([str(x) for x in result if x])
    return str(result)


def gemini_extract_structured(ocr_text: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None

    genai.configure(api_key=api_key)

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Sos un extractor de datos de comprobantes de pago (Argentina y Paraguay).
A partir del texto OCR, devolvé SOLO JSON válido (sin markdown) con estas claves:

{{
  "emitter": string|null,
  "recipient": string|null,
  "amount": number|null,
  "currency": "ARS"|"PYG"|null,
  "date": "YYYY-MM-DD"|null,
  "operation_id": string|null
}}

Reglas:
- Si hay varios montos, elegir el TOTAL o MONTO FINAL.
- currency: ARS o PYG según símbolos/pistas (₲, PYG, Guaraníes; ARS, Pesos, etc).
- date normalizada a YYYY-MM-DD si se puede.
- operation_id: ID de operación/transferencia/comprobante/autorización si existe.
- Si no estás seguro, usar null.

TEXTO OCR:
{ocr_text}
""".strip()

    resp = model.generate_content(prompt)
    text = getattr(resp, "text", "") or ""
    text = text.strip()

    # Intento de parseo robusto
    try:
        return json.loads(text)
    except Exception:
        # A veces el modelo mete texto extra: buscamos primer { ... }.
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def extract_all(image_bytes: bytes) -> ExtractedData:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ocr_text = easyocr_extract_text(img)

    # 1) Primero intentamos Gemini
    structured = gemini_extract_structured(ocr_text)

    data = ExtractedData(raw_text=ocr_text)

    if structured:
        data.emitter = structured.get("emitter")
        data.recipient = structured.get("recipient")
        data.amount = structured.get("amount")
        data.currency = structured.get("currency")
        data.date = structured.get("date")
        data.operation_id = structured.get("operation_id")
        data.extracted_json = json.dumps(structured, ensure_ascii=False)
        return data

    # 2) Fallback heurístico
    data.currency = _normalize_currency(ocr_text) or None
    data.amount = _parse_amount(ocr_text)
    data.date = _parse_date(ocr_text)
    data.operation_id = _parse_operation_id(ocr_text)
    data.emitter, data.recipient = _guess_parties(ocr_text)

    fallback_json = {
        "emitter": data.emitter,
        "recipient": data.recipient,
        "amount": data.amount,
        "currency": data.currency,
        "date": data.date,
        "operation_id": data.operation_id,
        "source": "heuristic_fallback",
        "extracted_at": datetime.utcnow().isoformat(),
    }
    data.extracted_json = json.dumps(fallback_json, ensure_ascii=False)
    return data
