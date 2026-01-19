# app.py
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from auth import ensure_default_admin, authenticate, create_worker, delete_user
from database import (
    SessionLocal,
    init_db,
    Receipt,
    User,
    sha256_exists,
    count_receipts_by_user,
)
from processor import sha256_bytes, extract_all


APP_TITLE = "UTARA - Gestión de Comprobantes"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Inicialización segura de sesión (CRÍTICO)
# =========================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "edit_rec_id" not in st.session_state:
    st.session_state.edit_rec_id = None
if "edit_payload" not in st.session_state:
    st.session_state.edit_payload = None


def set_page():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧾",
        layout="wide",
    )


def require_login():
    """
    Fuerza login: si no hay usuario autenticado, muestra el formulario.
    Si se autentica, setea st.session_state.auth_user y rerun.
    """
    if st.session_state.auth_user is None:
        st.title("🔐 Inicio de sesión")
        st.caption("Acceso restringido. Iniciá sesión para continuar.")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Usuario", placeholder="Tu usuario", autocomplete="username")
            password = st.text_input("Contraseña", type="password", autocomplete="current-password")
            submit = st.form_submit_button("Ingresar")

        if submit:
            user = authenticate(username.strip(), password)
            if not user:
                st.error("Credenciales inválidas.")
                st.stop()

            st.session_state.auth_user = user
            st.rerun()

        st.stop()

    return st.session_state.auth_user


def sidebar_nav(user):
    """
    Sidebar robusta: nunca asume user != None.
    Si no hay sesión, manda a Login (no crashea).
    """
    st.sidebar.title("🧭 Navegación")

    # 🔒 Protección absoluta
    if user is None:
        st.sidebar.warning("No hay sesión activa.")
        if st.sidebar.button("Ir a Login"):
            st.session_state.auth_user = None
            st.rerun()
        return "Login"

    st.sidebar.write(f"👤 **{user.username}**")
    st.sidebar.write(f"🔑 Rol: **{user.role}**")
    st.sidebar.divider()

    pages = ["Carga", "Historial"]
    if getattr(user, "role", "") == "admin":
        pages.append("Panel de Admin")

    page = st.sidebar.radio("Ir a:", pages, index=0)
    st.sidebar.divider()

    if st.sidebar.button("🚪 Cerrar sesión"):
        st.session_state.auth_user = None
        # limpiar editor si estaba abierto
        st.session_state.edit_rec_id = None
        st.session_state.edit_payload = None
        st.rerun()

    return page


def save_upload(bytes_data: bytes, original_name: str, sha256_hex: str) -> str:
    safe_name = "".join(c for c in original_name if c.isalnum() or c in ("-", "_", ".", " "))
    safe_name = safe_name.strip().replace(" ", "_")
    filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{sha256_hex[:12]}_{safe_name}"
    path = UPLOAD_DIR / filename
    path.write_bytes(bytes_data)
    return filename


def page_carga(user):
    # 🔒 si algo raro pasa, no crashear
    if user is None:
        st.warning("Debés iniciar sesión.")
        return

    st.header("📤 Carga de comprobantes")

    st.info(
        "Subí una imagen (JPG/PNG). El sistema detecta duplicados por SHA-256, hace OCR (EasyOCR) y estructura datos con Gemini (si hay API key)."
    )

    uploaded = st.file_uploader("Seleccioná un comprobante", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    img_bytes = uploaded.read()
    sha = sha256_bytes(img_bytes)

    with SessionLocal() as db:
        if sha256_exists(db, sha):
            st.error("🚫 Este comprobante ya fue cargado antes (duplicado detectado por SHA-256).")
            st.code(sha)
            return

    with st.spinner("Procesando (OCR + extracción)..."):
        data = extract_all(img_bytes)

    filename = save_upload(img_bytes, uploaded.name, sha)

    with SessionLocal() as db:
        rec = Receipt(
            user_id=user.id,
            image_filename=filename,
            image_sha256=sha,
            emitter=data.emitter,
            recipient=data.recipient,
            amount=data.amount,
            currency=data.currency,
            date=data.date,
            operation_id=data.operation_id,
            raw_text=data.raw_text,
            extracted_json=data.extracted_json,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(rec)
        db.commit()

    st.success("✅ Comprobante cargado y procesado.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Datos extraídos")
        st.json(
            {
                "emitter": data.emitter,
                "recipient": data.recipient,
                "amount": data.amount,
                "currency": data.currency,
                "date": data.date,
                "operation_id": data.operation_id,
                "sha256": sha,
                "file": filename,
            }
        )
    with col2:
        st.subheader("Texto OCR (debug)")
        st.text_area("OCR", value=(data.raw_text or "")[:8000], height=260)


def _fetch_receipts_for_user(db, user):
    if user.role == "admin":
        rows = db.query(Receipt).order_by(Receipt.created_at.desc()).all()
        return rows
    rows = (
        db.query(Receipt)
        .filter(Receipt.user_id == user.id)
        .order_by(Receipt.created_at.desc())
        .all()
    )
    return rows


def page_historial(user):
    if user is None:
        st.warning("Debés iniciar sesión.")
        return

    st.header("📚 Historial")

    with SessionLocal() as db:
        rows = _fetch_receipts_for_user(db, user)

    if not rows:
        st.warning("No hay registros todavía.")
        return

    df = pd.DataFrame([r.as_dict() for r in rows])

    if user.role == "admin":
        with SessionLocal() as db:
            users_map = {u.id: u.username for u in db.query(User).all()}
        df.insert(1, "username", df["user_id"].map(users_map))

    st.dataframe(
        df[
            [c for c in df.columns if c in (
                "id","username","user_id","emitter","recipient","amount","currency","date","operation_id","image_filename","created_at"
            )]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("🧾 Editar / Eliminar un registro")

    selected_id = st.number_input("ID del registro", min_value=1, step=1)

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("🔎 Cargar para editar"):
            with SessionLocal() as db:
                rec = db.query(Receipt).filter(Receipt.id == int(selected_id)).first()
                if not rec:
                    st.error("No existe ese ID.")
                    return
                if user.role != "admin" and rec.user_id != user.id:
                    st.error("No tenés permisos para editar este registro.")
                    return

                st.session_state.edit_rec_id = rec.id
                st.session_state.edit_payload = {
                    "emitter": rec.emitter or "",
                    "recipient": rec.recipient or "",
                    "amount": rec.amount if rec.amount is not None else "",
                    "currency": rec.currency or "",
                    "date": rec.date or "",
                    "operation_id": rec.operation_id or "",
                }
                st.rerun()

    with c2:
        if st.button("🗑️ Eliminar"):
            with SessionLocal() as db:
                rec = db.query(Receipt).filter(Receipt.id == int(selected_id)).first()
                if not rec:
                    st.error("No existe ese ID.")
                    return
                if user.role != "admin" and rec.user_id != user.id:
                    st.error("No tenés permisos para eliminar este registro.")
                    return

                try:
                    (UPLOAD_DIR / rec.image_filename).unlink(missing_ok=True)
                except Exception:
                    pass

                db.delete(rec)
                db.commit()

            st.success("✅ Registro eliminado.")
            time.sleep(0.5)
            st.rerun()

    # Editor
    if st.session_state.edit_rec_id is not None and st.session_state.edit_payload is not None:
        st.divider()
        st.subheader(f"✏️ Editando ID #{st.session_state.edit_rec_id}")

        payload = st.session_state.edit_payload or {}
        with st.form("edit_form"):
            emitter = st.text_input("Emisor", value=payload.get("emitter", ""))
            recipient = st.text_input("Destinatario", value=payload.get("recipient", ""))
            amount = st.text_input("Monto", value=str(payload.get("amount", "")))

            curr = payload.get("currency", "")
            if curr not in ["ARS", "PYG"]:
                curr = ""

            currency = st.selectbox("Moneda", ["", "ARS", "PYG"], index=["", "ARS", "PYG"].index(curr))
            date = st.text_input("Fecha (YYYY-MM-DD)", value=payload.get("date", ""))
            operation_id = st.text_input("ID Operación", value=payload.get("operation_id", ""))

            ok = st.form_submit_button("Guardar cambios")

        if ok:
            amount_val = None
            try:
                amount_val = float(str(amount).replace(",", ".").strip())
            except Exception:
                amount_val = None

            with SessionLocal() as db:
                rec = db.query(Receipt).filter(Receipt.id == int(st.session_state.edit_rec_id)).first()
                if not rec:
                    st.error("Registro no encontrado.")
                    return
                if user.role != "admin" and rec.user_id != user.id:
                    st.error("No tenés permisos.")
                    return

                rec.emitter = emitter.strip() or None
                rec.recipient = recipient.strip() or None
                rec.amount = amount_val
                rec.currency = currency.strip() or None
                rec.date = date.strip() or None
                rec.operation_id = operation_id.strip() or None
                rec.updated_at = datetime.utcnow()
                db.commit()

            st.success("✅ Cambios guardados.")
            st.session_state.edit_rec_id = None
            st.session_state.edit_payload = None
            time.sleep(0.4)
            st.rerun()

    st.divider()
    st.subheader("📦 Exportar a Excel")

    export_df = df.copy()
    preferred_cols = ["id", "username", "user_id", "emitter", "recipient", "amount", "currency", "date", "operation_id", "image_filename", "created_at", "updated_at"]
    export_df = export_df[[c for c in preferred_cols if c in export_df.columns]]

    file_name = f"comprobantes_{user.username}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
    towrite = _df_to_excel_bytes(export_df, sheet_name="Comprobantes")

    st.download_button(
        "⬇️ Descargar Excel",
        data=towrite,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        wb = writer.book
        ws = writer.sheets[sheet_name]

        header_fmt = wb.add_format({"bold": True})
        for col, name in enumerate(df.columns):
            ws.write(0, col, name, header_fmt)
            width = max(10, min(40, int(df[name].astype(str).str.len().max() if not df.empty else 10)))
            ws.set_column(col, col, width)

        ws.freeze_panes(1, 0)

    output.seek(0)
    return output.read()


def page_admin(user):
    if user is None:
        st.warning("Debés iniciar sesión.")
        return
    if user.role != "admin":
        st.error("No autorizado.")
        return

    st.header("🛠️ Panel de Admin")

    st.subheader("📊 Resumen de cargas por trabajador")
    with SessionLocal() as db:
        stats = count_receipts_by_user(db)
    st.dataframe(pd.DataFrame(stats, columns=["Usuario", "Cantidad de cargas"]), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("👥 Gestión de trabajadores")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ➕ Crear trabajador")
        with st.form("create_worker_form"):
            new_user = st.text_input("Nuevo usuario", placeholder="ej: worker_juan")
            new_pass = st.text_input("Contraseña", type="password")
            ok = st.form_submit_button("Crear")
        if ok:
            try:
                if not new_user.strip() or not new_pass.strip():
                    raise ValueError("Usuario y contraseña son obligatorios.")
                create_worker(new_user.strip(), new_pass)
                st.success("✅ Trabajador creado.")
            except Exception as e:
                st.error(str(e))

    with c2:
        st.markdown("### 🗑️ Eliminar usuario")
        with st.form("delete_user_form"):
            del_user = st.text_input("Usuario a eliminar", placeholder="ej: worker_juan")
            ok2 = st.form_submit_button("Eliminar")
        if ok2:
            try:
                if not del_user.strip():
                    raise ValueError("Indicá un usuario.")
                delete_user(del_user.strip())
                st.success("✅ Usuario eliminado.")
            except Exception as e:
                st.error(str(e))


def main():
    set_page()

    # DB + admin por defecto
    init_db()
    ensure_default_admin()

    # user seguro
    user = st.session_state.get("auth_user")
    if user is None:
        user = require_login()

    page = sidebar_nav(user)

    st.title("🧾 " + APP_TITLE)

    if page == "Carga":
        page_carga(user)
    elif page == "Historial":
        page_historial(user)
    elif page == "Panel de Admin":
        page_admin(user)
    elif page == "Login":
        # fallback por si algo raro pasó
        st.session_state.auth_user = None
        st.rerun()


if __name__ == "__main__":
    main()
