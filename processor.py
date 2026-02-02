# processor.py
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import easyocr
import requests

# =========================
# PERF / THREADS (CPU)
# =========================
# Reduce “sobre-threading” en containers chicos (Railway)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# =========================
# EASYOCR (init 1 sola vez)
# =========================
# verbose=False reduce logs y un poquito overhead
EASYOCR_READER = easyocr.Reader(["es"], gpu=False, verbose=False)


# =========================
# MODELO DE DATOS
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
# UTILIDADES
# =========================
def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _clean_spaces(s: str) -> str:
    return " ".join((s or "").strip().split())


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

    money_candidates = re.findall(r"\$\s*([\d\.,]+)", text)
    candidates = (
        money_candidates
        if money_candidates
        else re.findall(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)(?!\d)", text)
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

    m = re.search(
        r"\b(\d{1,2})\s+de\s+([a-záéíóúñ]+)\s+de\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if m:
        months = {
            "enero": "01",
            "febrero": "02",
            "marzo": "03",
            "abril": "04",
            "mayo": "05",
            "junio": "06",
            "julio": "07",
            "agosto": "08",
            "septiembre": "09",
            "setiembre": "09",
            "octubre": "10",
            "noviembre": "11",
            "diciembre": "12",
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

    m2 = re.search(
        r"N[ÚU]MERO\s+DE\s+OPERACI[ÓO]N(?:\s+DE\s+MERCADO\s+PAGO)?",
        text,
        re.IGNORECASE,
    )
    if m2:
        tail = text[m2.end() :]
        m3 = re.search(r"\b(\d{6,})\b", tail)
        if m3:
            return m3.group(1)

    return None


# =========================
# EMISOR / RECEPTOR (De / Para)
# =========================
def _split_lines(text: str) -> List[str]:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def _extract_name_after_label(lines: List[str], label: str) -> Optional[str]:
    label_re = re.compile(rf"^{re.escape(label)}$", re.IGNORECASE)

    for i, line in enumerate(lines):
        if label_re.fullmatch(line.strip()):
            if i + 1 < len(lines) and _looks_like_name(lines[i + 1]):
                return _clean_spaces(lines[i + 1])

        m = re.match(rf"^({label})\s+(.+)$", line, re.IGNORECASE)
        if m:
            cand = _clean_spaces(m.group(2))
            if _looks_like_name(cand):
                return cand

    return None


def _extract_two_people_by_cuit(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    person_pattern = re.compile(
        r"^([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+)+)\s+.*?(CUIT|CUIL|CUIT/CUIL|CUITCUIL|CUITCUIl)[:\s]*\d{2}[- ]?\d{8}[- ]?\d\b",
        re.IGNORECASE,
    )

    persons = []
    for line in lines:
        line2 = (
            line.replace("Cuitcuil", "CUIT/CUIL")
            .replace("CuitCUIL", "CUIT/CUIL")
            .replace("CuitCuil", "CUIT/CUIL")
        )
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
# OCR: PREPROCESADO + RECORTES (ACELERA MUCHO)
# =========================
def _preprocess(img: Image.Image) -> Image.Image:
    # Grayscale + autocontrast + sharpen suave
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = ImageEnhance.Sharpness(g).enhance(1.6)
    return g


def _resize_max_width(img: Image.Image, max_width: int) -> Image.Image:
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    return img.resize((max_width, int(img.height * ratio)))


def _crop_rel(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))


def _ocr_image(img: Image.Image) -> str:
    arr = np.array(img)
    text = "\n".join(EASYOCR_READER.readtext(arr, detail=0, paragraph=True))
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    text = re.sub(r"\bhs\b", "", text, flags=re.IGNORECASE)
    text = (
        text.replace("Cuitcuil", "CUIT/CUIL")
        .replace("CuitCUIL", "CUIT/CUIL")
        .replace("CuitCuil", "CUIT/CUIL")
    )
    return text


def easyocr_extract_text_fast(image: Image.Image) -> Tuple[str, str]:
    """
    Devuelve: (text_quick, text_full_if_needed)

    Estrategia:
    - OCR rápido por recortes (zona De/Para + zona Monto + zona Operación)
    - SOLO si faltan campos, hace OCR completo (más lento)
    """
    # Mucho más rápido en CPU
    image = _resize_max_width(image, max_width=900)
    image = _preprocess(image)

    # Recortes típicos para MercadoPago (pantalla)
    crops = [
        ("people", _crop_rel(image, 0.06, 0.35, 0.94, 0.75)),
        ("amount",  _crop_rel(image, 0.05, 0.18, 0.95, 0.40)),
        ("op",      _crop_rel(image, 0.05, 0.72, 0.95, 0.98)),
    ]

    quick_parts = []
    for _, c in crops:
        quick_parts.append(_ocr_image(c))

    quick_text = "\n".join([p for p in quick_parts if p.strip()])

    # Chequeo rápido: si ya tenemos lo esencial, evitamos OCR completo
    em, rc = extract_emitter_recipient_from_text(quick_text)
    amt = _parse_amount(quick_text)

    if (em and rc and amt):
        return quick_text, ""  # full no necesario

    # OCR completo solo si faltan datos
    full_text = _ocr_image(image)
    return quick_text, full_text


# =========================
# GEMINI (solo si hace falta)
# =========================
def gemini_extract_structured(ocr_text: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    # Si querés forzar desactivar Gemini (para máxima velocidad):
    # export GEMINI_DISABLED=1
    if os.getenv("GEMINI_DISABLED", "0") == "1":
        return None

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"""
Extraé datos de un comprobante (Mercado Pago / Argentina / Paraguay).
Respondé SOLO JSON válido.

Reglas:
- amount: monto principal (priorizar el que tenga $ o sea el más grande).
- Emisor/Receptor: suelen estar en "De" y "Para".
- operation_id: suele estar en "Número de operación".
- NO inventar.

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
                        }
                    ]
                }
            ]
        }

        r = requests.post(
            f"{url}?key={api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=18,  # menor timeout para no colgar
        )
        if r.status_code != 200:
            return None

        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except Exception:
        return None


# =========================
# PIPELINE
# =========================
def extract_all(image_bytes: bytes) -> ExtractedData:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    quick_text, full_text = easyocr_extract_text_fast(image)

    # Texto final para debug (preferimos full si existe)
    ocr_text = (full_text or quick_text or "").strip()
    data = ExtractedData(raw_text=ocr_text)

    # Reglas locales (rápidas)
    rule_emitter, rule_recipient = extract_emitter_recipient_from_text(ocr_text)
    rule_amount = _parse_amount(ocr_text)
    rule_currency = _normalize_currency(ocr_text)
    rule_date = _parse_date(ocr_text)
    rule_operation_id = _parse_operation_id(ocr_text)

    data.emitter = rule_emitter
    data.recipient = rule_recipient
    data.amount = rule_amount
    data.currency = rule_currency
    data.date = rule_date
    data.operation_id = rule_operation_id

    # Gemini SOLO si falta algo importante (para acelerar)
    need_ai = (data.emitter is None or data.recipient is None or data.amount is None)
    if need_ai:
        structured = gemini_extract_structured(ocr_text)
        if structured:
            data.emitter = structured.get("emitter") or data.emitter
            data.recipient = structured.get("recipient") or data.recipient
            if structured.get("amount") is not None:
                data.amount = structured.get("amount")
            data.currency = structured.get("currency") or data.currency
            data.date = structured.get("date") or data.date
            data.operation_id = structured.get("operation_id") or data.operation_id
            data.extracted_json = json.dumps(structured, ensure_ascii=False)
            return data

    data.extracted_json = json.dumps(
        {"source": "rules_fast", "raw_text": ocr_text},
        ensure_ascii=False,
    )
    return data
