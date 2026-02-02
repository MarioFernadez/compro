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

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Session init
# =========================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "edit_rec_id" not in st.session_state:
    st.session_state.edit_rec_id = None


def set_page():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")


def require_login():
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
    st.sidebar.title("🧭 Navegación")

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
        st.session_state.edit_rec_id = None
        st.rerun()

    return page


def save_upload(bytes_data: bytes, original_name: str, sha256_hex: str) -> str:
    safe_name = "".join(c for c in original_name if c.isalnum() or c in ("-", "_", ".", " "))
    safe_name = safe_name.strip().replace(" ", "_")
    filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{sha256_hex[:12]}_{safe_name}"
    (UPLOAD_DIR / filename).write_bytes(bytes_data)
    return filename


def _fetch_receipts_for_user(db, user):
    if user.role == "admin":
        return db.query(Receipt).order_by(Receipt.created_at.desc()).all()

    return (
        db.query(Receipt)
        .filter(Receipt.user_id == user.id)
        .order_by(Receipt.created_at.desc())
        .all()
    )


def _shorten(s: str, n: int = 48) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else (s[: n - 1] + "…")


# ✅ Cache afuera
@st.cache_data(show_spinner=False)
def _cached_extract(sha: str, img_bytes: bytes):
    return extract_all(img_bytes)


def page_carga(user):
    if user is None:
        st.warning("Debés iniciar sesión.")
        return

    st.header("📤 Carga de comprobantes")

    st.info(
        "Subí imágenes (JPG/PNG). Se detectan duplicados por SHA-256. "
        "OCR optimizado para Mercado Pago."
    )

    uploads = st.file_uploader(
        "Seleccioná comprobantes",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if not uploads:
        return

    start_btn = st.button("🚀 Procesar archivos", use_container_width=True)
    if not start_btn:
        st.caption("Cuando termines de seleccionar, tocá **Procesar archivos**.")
        return

    progress = st.progress(0)
    status = st.empty()

    total = len(uploads)
    ok_count = 0
    dup_count = 0
    fail_count = 0

    for idx, uploaded in enumerate(uploads, start=1):
        try:
            status.write(f"Procesando {idx}/{total}: **{uploaded.name}**")
            img_bytes = uploaded.read()
            sha = sha256_bytes(img_bytes)

            # 1) duplicados (DB)
            with SessionLocal() as db:
                if sha256_exists(db, sha):
                    dup_count += 1
                    progress.progress(int(idx / total * 100))
                    continue

            # 2) OCR + extracción
            data = _cached_extract(sha, img_bytes)

            # 3) guardar archivo + DB
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

            ok_count += 1

        except Exception as e:
            fail_count += 1
            st.error(f"Error procesando {uploaded.name}: {e}")

        progress.progress(int(idx / total * 100))

    status.empty()
    st.success(f"✅ Listo. Cargados: {ok_count} | Duplicados: {dup_count} | Fallidos: {fail_count}")


def _can_access(user, rec: Receipt) -> bool:
    if user is None or rec is None:
        return False
    return user.role == "admin" or rec.user_id == user.id


def _delete_receipt_and_file(db, rec: Receipt) -> tuple[bool, str]:
    try:
        # borrar archivo
        if rec.image_filename:
            try:
                (UPLOAD_DIR / rec.image_filename).unlink(missing_ok=True)
            except Exception:
                pass

        db.delete(rec)
        db.commit()
        return True, "✅ Comprobante eliminado."
    except Exception as e:
        db.rollback()
        return False, f"❌ Error eliminando: {e}"


def _update_receipt(db, rec: Receipt, emitter, recipient, amount, currency, date, operation_id) -> tuple[bool, str]:
    try:
        rec.emitter = emitter or None
        rec.recipient = recipient or None

        # amount puede venir vacío -> None
        if amount is None or str(amount).strip() == "":
            rec.amount = None
        else:
            rec.amount = float(amount)

        rec.currency = currency or None
        rec.date = date or None
        rec.operation_id = operation_id or None
        rec.updated_at = datetime.utcnow()

        db.add(rec)
        db.commit()
        return True, "✅ Cambios guardados."
    except Exception as e:
        db.rollback()
        return False, f"❌ Error guardando: {e}"


def page_historial(user):
    if user is None:
        st.warning("Debés iniciar sesión.")
        return

    st.header("📚 Historial")

    st.markdown(
        """
        <style>
          div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
          div[data-testid="stDataFrame"] * { font-size: 13px; }
          div[data-testid="stDataFrame"] td { white-space: nowrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    if "image_filename" in df.columns:
        df["archivo"] = df["image_filename"].apply(lambda x: _shorten(x, 55))
    else:
        df["archivo"] = ""

    cols = ["id"]
    if "username" in df.columns:
        cols += ["username"]
    cols += [
        "user_id",
        "emitter",
        "recipient",
        "amount",
        "currency",
        "date",
        "operation_id",
        "archivo",
        "created_at",
    ]
    cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "username": st.column_config.TextColumn("Usuario", width="medium"),
            "user_id": st.column_config.NumberColumn("User ID", width="small"),
            "emitter": st.column_config.TextColumn("Emisor", width="medium"),
            "recipient": st.column_config.TextColumn("Receptor", width="medium"),
            "amount": st.column_config.NumberColumn("Monto", format="%.2f", width="small"),
            "currency": st.column_config.TextColumn("Moneda", width="small"),
            "date": st.column_config.TextColumn("Fecha", width="small"),
            "operation_id": st.column_config.TextColumn("Operación", width="small"),
            "archivo": st.column_config.TextColumn("Archivo", width="large"),
            "created_at": st.column_config.TextColumn("Creado", width="medium"),
        },
    )

    st.divider()
    st.subheader("📷 Ver comprobante")

    ids = sorted(df["id"].dropna().astype(int).unique().tolist())
    sel_id = st.selectbox("Seleccioná un ID", ids, index=0, key="sel_id")

    with SessionLocal() as db:
        rec = db.query(Receipt).filter(Receipt.id == int(sel_id)).first()

    if not rec:
        st.error("No se encontró el registro.")
        return

    if not _can_access(user, rec):
        st.error("No tenés permisos para ver este registro.")
        return

    img_path = UPLOAD_DIR / (rec.image_filename or "")

    colA, colB = st.columns([2, 1])

    with colA:
        if img_path.exists():
            try:
                img_bytes = img_path.read_bytes()
                if img_bytes:
                    st.image(img_bytes, caption=rec.image_filename, use_column_width=True)
                else:
                    st.warning("La imagen existe pero está vacía (0 bytes).")
            except Exception as e:
                st.error("No se pudo abrir la imagen.")
                st.code(str(e))
        else:
            st.warning("No se encontró la imagen en disco.")
            st.caption(f"Ruta esperada: {img_path}")

    with colB:
        st.write("**Datos**")
        st.write(f"- Emisor: {rec.emitter or '-'}")
        st.write(f"- Receptor: {rec.recipient or '-'}")
        st.write(f"- Monto: {rec.amount if rec.amount is not None else '-'} {rec.currency or ''}")
        st.write(f"- Fecha: {rec.date or '-'}")
        st.write(f"- Operación: {rec.operation_id or '-'}")
        st.write(f"- SHA256: {rec.image_sha256[:16] + '…' if rec.image_sha256 else '-'}")
        st.write(f"- Archivo: {rec.image_filename or '-'}")

        if img_path.exists():
            ext = img_path.suffix.lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png" if ext == ".png" else "application/octet-stream"
            st.download_button(
                "⬇️ Descargar comprobante",
                data=img_path.read_bytes(),
                file_name=rec.image_filename,
                mime=mime,
            )

    # ======================================================
    # ✅ EDITAR
    # ======================================================
    st.divider()
    st.subheader("✏️ Editar comprobante")

    with st.form("edit_receipt_form", clear_on_submit=False):
        emitter = st.text_input("Emisor", value=rec.emitter or "")
        recipient = st.text_input("Receptor", value=rec.recipient or "")

        c1, c2 = st.columns(2)
        with c1:
            amount = st.text_input("Monto", value="" if rec.amount is None else str(rec.amount))
        with c2:
            currency = st.selectbox(
                "Moneda",
                options=["", "ARS", "PYG", "USD"],
                index=(["", "ARS", "PYG", "USD"].index(rec.currency) if rec.currency in ["ARS", "PYG", "USD"] else 0),
            )

        c3, c4 = st.columns(2)
        with c3:
            date = st.text_input("Fecha (texto o YYYY-MM-DD)", value=rec.date or "")
        with c4:
            operation_id = st.text_input("Operación", value=rec.operation_id or "")

        save_btn = st.form_submit_button("💾 Guardar cambios")

    if save_btn:
        with SessionLocal() as db:
            rec_db = db.query(Receipt).filter(Receipt.id == int(sel_id)).first()
            if not rec_db:
                st.error("No existe el registro.")
                st.stop()
            if not _can_access(user, rec_db):
                st.error("No tenés permisos para editar este registro.")
                st.stop()

            ok, msg = _update_receipt(
                db,
                rec_db,
                emitter=emitter.strip(),
                recipient=recipient.strip(),
                amount=amount,
                currency=currency.strip(),
                date=date.strip(),
                operation_id=operation_id.strip(),
            )

        if ok:
            st.success(msg)
            time.sleep(0.25)
            st.rerun()
        else:
            st.error(msg)

    # ======================================================
    # ✅ ELIMINAR 1 POR 1
    # ======================================================
    st.divider()
    st.subheader("🗑️ Eliminar comprobante")

    st.warning("⚠️ Eliminar borra el registro y también el archivo de imagen. No se puede deshacer.")
    confirm_del = st.checkbox(f"Confirmo eliminar el comprobante ID {sel_id}", key="confirm_del")

    if st.button("🗑️ Eliminar este comprobante", type="primary", disabled=not confirm_del):
        with SessionLocal() as db:
            rec_db = db.query(Receipt).filter(Receipt.id == int(sel_id)).first()
            if not rec_db:
                st.error("No existe el registro.")
                st.stop()
            if not _can_access(user, rec_db):
                st.error("No tenés permisos para eliminar este registro.")
                st.stop()

            ok, msg = _delete_receipt_and_file(db, rec_db)

        if ok:
            st.cache_data.clear()
            st.success(msg)
            time.sleep(0.25)
            st.rerun()
        else:
            st.error(msg)

    # ======================================================
    # EXPORT
    # ======================================================
    st.divider()
    st.subheader("📦 Exportar a Excel")

    export_df = df.copy()
    preferred_cols = [
        "id",
        "username",
        "user_id",
        "emitter",
        "recipient",
        "amount",
        "currency",
        "date",
        "operation_id",
        "image_filename",
        "created_at",
        "updated_at",
    ]
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
    st.dataframe(
        pd.DataFrame(stats, columns=["Usuario", "Cantidad de cargas"]),
        hide_index=True,
        use_container_width=True,
    )

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

    init_db()
    ensure_default_admin()

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
        st.session_state.auth_user = None
        st.rerun()


if __name__ == "__main__":
    main()
