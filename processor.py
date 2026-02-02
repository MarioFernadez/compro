# processor.py
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import easyocr
import requests

# =========================
# Performance (CPU)
# =========================
# Evita que torch se coma todo el CPU y te "cuelgue" Railway
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

# =========================
# EasyOCR global (1 sola vez)
# =========================
EASYOCR_READER = easyocr.Reader(["es"], gpu=False)

# =========================
# Data model
# =========================
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


# =========================
# Utils
# =========================
def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _clean_spaces(s: str) -> str:
    return " ".join((s or "").strip().split())


def _split_lines(text: str):
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def _looks_like_name(s: str) -> bool:
    s = _clean_spaces(s)
    if not s or len(s) < 5:
        return False
    if re.search(r"\b(CUIT|CUIL|CVU|CBU|BANCO|MERCADO\s+PAGO)\b", s, re.IGNORECASE):
        return False
    return len(s.split()) >= 2


def _normalize_currency(text: str) -> Optional[str]:
    t = (text or "").upper()
    if any(x in t for x in ["PYG", "GUARANI", "GUARANÍ", "GS", "₲"]):
        return "PYG"
    if any(x in t for x in ["ARS", "ARG", "PESO", "$"]):
        return "ARS"
    return None


def _parse_amount(text: str) -> Optional[float]:
    if not text:
        return None

    # Primero montos con $
    money_candidates = re.findall(r"\$\s*([\d\.,]+)", text)
    candidates = money_candidates if money_candidates else re.findall(
        r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)(?!\d)", text
    )

    def to_float(s: str) -> Optional[float]:
        try:
            s = s.replace(" ", "").strip()
            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
            else:
                if re.match(r"^\d{1,3}(\.\d{3})+$", s):
                    s = s.replace(".", "")
                elif re.match(r"^\d{1,3}(,\d{3})+$", s):
                    s = s.replace(",", "")
                else:
                    s = s.replace(",", ".")
            v = float(s)
            return v if v > 0 else None
        except Exception:
            return None

    values = [to_float(c) for c in candidates]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _parse_date(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"\b(\d{1,2})\s+de\s+([a-záéíóúñ]+)\s+de\s+(\d{4})\b", text, re.IGNORECASE)
    if m:
        months = {
            "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
            "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
            "septiembre": "09", "setiembre": "09",
            "octubre": "10", "noviembre": "11", "diciembre": "12",
        }
        day = str(m.group(1)).zfill(2)
        month = months.get(m.group(2).lower(), "01")
        year = m.group(3)
        return f"{year}-{month}-{day}"

    return None


def _parse_operation_id(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"N[ÚU]MERO\s+DE\s+OPERACI[ÓO]N(?:\s+DE\s+MERCADO\s+PAGO)?\s*[:\-]?\s*(\d{6,})",
        r"NUMERO\s+DE\s+OPERACION(?:\s+DE\s+MERCADO\s+PAGO)?\s*[:\-]?\s*(\d{6,})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)

    m2 = re.search(r"N[ÚU]MERO\s+DE\s+OPERACI[ÓO]N(?:\s+DE\s+MERCADO\s+PAGO)?", text, re.IGNORECASE)
    if m2:
        tail = text[m2.end():]
        m3 = re.search(r"\b(\d{6,})\b", tail)
        if m3:
            return m3.group(1)

    return None


# =========================
# Emisor / Receptor ("De" / "Para")
# =========================
def _extract_name_after_label(lines, label: str) -> Optional[str]:
    # Busca "De" o "Para" como línea sola, o "De Nombre"
    for i, line in enumerate(lines):
        if line.strip().lower() == label.lower():
            if i + 1 < len(lines) and _looks_like_name(lines[i + 1]):
                return _clean_spaces(lines[i + 1])

        m = re.match(rf"^{label}\s+(.+)$", line, re.IGNORECASE)
        if m:
            cand = _clean_spaces(m.group(1))
            if _looks_like_name(cand):
                return cand
    return None


def _extract_two_people_by_cuit(lines) -> Tuple[Optional[str], Optional[str]]:
    # Fallback: primera persona con CUIT/CUIL = emisor, segunda = receptor
    person_pattern = re.compile(
        r"^([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+)+).*(CUIT|CUIL|CUIT/CUIL).*(\d{2}[- ]?\d{8}[- ]?\d)\b",
        re.IGNORECASE,
    )

    persons = []
    for line in lines:
        line2 = line.replace("Cuitcuil", "CUIT/CUIL").replace("CuitCUIL", "CUIT/CUIL").replace("CuitCuil", "CUIT/CUIL")
        m = person_pattern.search(line2)
        if m:
            name = _clean_spaces(m.group(1))
            if _looks_like_name(name):
                persons.append(name)

    if len(persons) >= 2:
        return persons[0], persons[1]
    return None, None


def extract_emitter_recipient_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    lines = _split_lines(text)
    emitter = _extract_name_after_label(lines, "De")
    recipient = _extract_name_after_label(lines, "Para")

    if emitter or recipient:
        return emitter, recipient

    return _extract_two_people_by_cuit(lines)


# =========================
# Image preprocessing
# =========================
def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Grayscale + autocontrast + sharpen leve (rápido y útil para fotos de celular)
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = ImageEnhance.Sharpness(g).enhance(1.6)
    return g


# =========================
# OCR
# =========================
def easyocr_extract_text(image: Image.Image, max_width: int) -> str:
    img = image.copy()

    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))

    img = _preprocess_for_ocr(img)

    text = "\n".join(
        EASYOCR_READER.readtext(np.array(img), detail=0, paragraph=True)
    )

    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    text = re.sub(r"\bhs\b", "", text, flags=re.IGNORECASE)

    text = text.replace("Cuitcuil", "CUIT/CUIL").replace("CuitCUIL", "CUIT/CUIL").replace("CuitCuil", "CUIT/CUIL")
    return text


# =========================
# Gemini (opcional)
# =========================
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
Extraé datos de un comprobante (Mercado Pago / Argentina / Paraguay).
Respondé SOLO JSON válido.

Reglas ESTRICTAS:
- amount: monto monetario principal (priorizar el que tenga $ o sea el más grande).
- NO inventar datos.
- Si el comprobante trae secciones "De" y "Para", ahí están emisor y receptor.
- operation_id suele estar en "Número de operación".

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
            timeout=18,
        )
        if r.status_code != 200:
            return None

        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)

    except Exception:
        return None


# =========================
# Main pipeline (FAST)
# =========================
def extract_all(image_bytes: bytes) -> ExtractedData:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    data = ExtractedData()

    # 1) OCR rápido
    ocr_fast = easyocr_extract_text(image, max_width=1024)

    # reglas rápidas
    em, rec = extract_emitter_recipient_from_text(ocr_fast)
    amt = _parse_amount(ocr_fast)
    cur = _normalize_currency(ocr_fast)
    dte = _parse_date(ocr_fast)
    opid = _parse_operation_id(ocr_fast)

    # Si faltan personas, hacemos OCR "lento" (mejor calidad)
    if not em or not rec:
        ocr_slow = easyocr_extract_text(image, max_width=1600)
        em2, rec2 = extract_emitter_recipient_from_text(ocr_slow)
        amt2 = _parse_amount(ocr_slow) or amt
        cur2 = _normalize_currency(ocr_slow) or cur
        dte2 = _parse_date(ocr_slow) or dte
        opid2 = _parse_operation_id(ocr_slow) or opid

        # merge
        em = em or em2
        rec = rec or rec2
        amt = amt2
        cur = cur2
        dte = dte2
        opid = opid2

        data.raw_text = ocr_slow
    else:
        data.raw_text = ocr_fast

    # (Opcional) Gemini solo si querés "mejorar" y hay API, pero puede tardar
    structured = gemini_extract_structured(data.raw_text or "")
    if structured:
        data.emitter = structured.get("emitter") or em
        data.recipient = structured.get("recipient") or rec
        data.amount = structured.get("amount") if structured.get("amount") is not None else amt
        data.currency = structured.get("currency") or cur
        data.date = structured.get("date") or dte
        data.operation_id = structured.get("operation_id") or opid
        data.extracted_json = json.dumps(structured, ensure_ascii=False)
        return data

    # reglas
    data.emitter = em
    data.recipient = rec
    data.amount = amt
    data.currency = cur
    data.date = dte
    data.operation_id = opid
    data.extracted_json = json.dumps({"source": "rules", "raw_text": data.raw_text}, ensure_ascii=False)
    return data
