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


def _clean_spaces(s: str) -> str:
    return " ".join((s or "").strip().split())


def _looks_like_name(s: str) -> bool:
    s = _clean_spaces(s)
    if not s or len(s) < 5:
        return False
    # evita agarrar líneas tipo "CUIT/CUIL: 27-..."
    if re.search(r"\b(CUIT|CUIL|CVU|CBU|BANCO|MERCADO\s+PAGO)\b", s, re.IGNORECASE):
        return False
    # debe tener al menos 2 palabras
    return len(s.split()) >= 2


def _normalize_currency(text: str) -> Optional[str]:
    t = (text or "").upper()
    if any(x in t for x in ["PYG", "GUARANI", "GUARANÍ", "GS", "₲"]):
        return "PYG"
    if any(x in t for x in ["ARS", "ARG", "PESO", "$"]):
        return "ARS"
    return None


def _parse_amount(text: str) -> Optional[float]:
    """
    Soporta:
    - $ 124.740
    - 124.740
    - 124,740 (OCR raro)
    - 124740
    - 42.000 / 42,000 / 42.000,00 etc
    """
    if not text:
        return None

    # Preferir montos con $
    money_candidates = re.findall(r"\$\s*([\d\.,]+)", text)
    candidates = money_candidates if money_candidates else re.findall(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)(?!\d)", text)

    def to_float(s: str) -> Optional[float]:
        try:
            s = s.replace(" ", "").strip()
            # si tiene ambos separadores, decidir por el último como decimal
            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
            else:
                # si tiene un solo separador:
                # si parece miles (xxx.xxx) sin decimales -> quitar separador
                if re.match(r"^\d{1,3}(\.\d{3})+$", s):
                    s = s.replace(".", "")
                elif re.match(r"^\d{1,3}(,\d{3})+$", s):
                    s = s.replace(",", "")
                else:
                    # puede ser decimal con coma
                    s = s.replace(",", ".")
            return float(s)
        except Exception:
            return None

    values = [to_float(c) for c in candidates]
    values = [v for v in values if v is not None and v > 0]
    return max(values) if values else None


def _parse_date(text: str) -> Optional[str]:
    """
    Soporta:
    - "05 de enero de 2026"
    - "Lunes, 05 de enero de 2026 a las ..."
    """
    if not text:
        return None

    m = re.search(r"\b(\d{1,2})\s+de\s+([a-záéíóúñ]+)\s+de\s+(\d{4})\b", text, re.IGNORECASE)
    if m:
        months = {
            "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
            "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
            "septiembre": "09", "setiembre": "09",  # OCR suele variar
            "octubre": "10", "noviembre": "11", "diciembre": "12",
        }
        day = str(m.group(1)).zfill(2)
        month = months.get(m.group(2).lower(), "01")
        year = m.group(3)
        return f"{year}-{month}-{day}"

    return None


def _parse_operation_id(text: str) -> Optional[str]:
    """
    Mercado Pago suele traer:
    - "Número de operación de Mercado Pago"
    y debajo un número.
    """
    if not text:
        return None

    # Caso 1: número en misma línea o siguiente
    patterns = [
        r"N[ÚU]MERO\s+DE\s+OPERACI[ÓO]N(?:\s+DE\s+MERCADO\s+PAGO)?\s*[:\-]?\s*(\d{6,})",
        r"NUMERO\s+DE\s+OPERACION(?:\s+DE\s+MERCADO\s+PAGO)?\s*[:\-]?\s*(\d{6,})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)

    # Caso 2: detecta el bloque y agarra el primer número largo después
    m2 = re.search(r"N[ÚU]MERO\s+DE\s+OPERACI[ÓO]N(?:\s+DE\s+MERCADO\s+PAGO)?", text, re.IGNORECASE)
    if m2:
        tail = text[m2.end():]
        m3 = re.search(r"\b(\d{6,})\b", tail)
        if m3:
            return m3.group(1)

    return None


# =====================================================
# EXTRA: EMISOR / RECEPTOR Mercado Pago (De / Para)
# =====================================================

def _split_lines(text: str):
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def _extract_name_after_label(lines, label_regex: re.Pattern) -> Optional[str]:
    """
    Busca línea que sea "De" o "Para" y retorna el nombre en la misma línea
    o en la siguiente línea.
    """
    for i, line in enumerate(lines):
        if label_regex.fullmatch(line.strip()):
            # nombre suele estar justo abajo
            if i + 1 < len(lines) and _looks_like_name(lines[i + 1]):
                return _clean_spaces(lines[i + 1])
        # a veces OCR junta: "De Sandra Gabriela Diaz"
        m = re.match(r"^(De|Para)\s+(.+)$", line, re.IGNORECASE)
        if m:
            cand = _clean_spaces(m.group(2))
            if _looks_like_name(cand):
                return cand
    return None


def _extract_two_people_by_cuit(lines) -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback: si OCR no detecta bien "De/Para" pero hay dos personas con CUIT/CUIL.
    Toma primera como emisor y segunda como receptor.
    """
    person_pattern = re.compile(
        r"^([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+)+)\s+.*?(CUIT|CUIL|CUIT/CUIL|CUITCUIL|CUITCUIl)[:\s]*\d{2}[- ]?\d{8}[- ]?\d\b",
        re.IGNORECASE,
    )

    persons = []
    for line in lines:
        line2 = line.replace("Cuitcuil", "CUIT/CUIL").replace("CuitCUIL", "CUIT/CUIL")
        m = person_pattern.search(line2)
        if m:
            name = _clean_spaces(m.group(1))
            if _looks_like_name(name):
                persons.append(name)

    if len(persons) >= 2:
        return persons[0], persons[1]
    return None, None


def extract_emitter_recipient_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Estrategia:
    1) Buscar labels "De" y "Para"
    2) Si no, buscar 2 personas con CUIT/CUIL
    """
    lines = _split_lines(text)

    # 1) "De" y "Para"
    emitter = _extract_name_after_label(lines, re.compile(r"^De$", re.IGNORECASE))
    recipient = _extract_name_after_label(lines, re.compile(r"^Para$", re.IGNORECASE))

    if emitter or recipient:
        return emitter, recipient

    # 2) fallback por CUIT/CUIL
    return _extract_two_people_by_cuit(lines)


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

    # 🔒 LIMPIEZA CLAVE: eliminar horarios (no son IDs ni montos)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    text = re.sub(r"\bhs\b", "", text, flags=re.IGNORECASE)

    # normalizar variantes comunes de OCR
    text = text.replace("Cuitcuil", "CUIT/CUIL").replace("CuitCUIL", "CUIT/CUIL").replace("CuitCuil", "CUIT/CUIL")
    return text


# =====================================================
# GEMINI VIA REST
# =====================================================

def gemini_extract_structured(ocr_text: str) -> Optional[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # silencioso para producción
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
            timeout=30,
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
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    ocr_text = easyocr_extract_text(image)
    data = ExtractedData(raw_text=ocr_text)

    # ✅ Reglas locales primero (rápido y estable)
    rule_emitter, rule_recipient = extract_emitter_recipient_from_text(ocr_text)
    rule_amount = _parse_amount(ocr_text)
    rule_currency = _normalize_currency(ocr_text)
    rule_date = _parse_date(ocr_text)
    rule_operation_id = _parse_operation_id(ocr_text)

    # Luego Gemini (si existe)
    structured = gemini_extract_structured(ocr_text)

    if structured:
        data.emitter = structured.get("emitter") or rule_emitter
        data.recipient = structured.get("recipient") or rule_recipient
        data.amount = structured.get("amount") if structured.get("amount") is not None else rule_amount
        data.currency = structured.get("currency") or rule_currency
        data.date = structured.get("date") or rule_date
        data.operation_id = structured.get("operation_id") or rule_operation_id
        data.extracted_json = json.dumps(structured, ensure_ascii=False)
        return data

    # fallback 100% reglas
    data.emitter = rule_emitter
    data.recipient = rule_recipient
    data.amount = rule_amount
    data.currency = rule_currency
    data.date = rule_date
    data.operation_id = rule_operation_id

    data.extracted_json = json.dumps(
        {"source": "rules_fallback", "raw_text": ocr_text},
        ensure_ascii=False,
    )

    return data
