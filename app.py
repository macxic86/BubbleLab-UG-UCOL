import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from scipy import ndimage as ndi

# === POSTHOG METRICS ===
import os
import uuid
from posthog import Posthog

POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")

if POSTHOG_API_KEY:
    posthog = Posthog(project_api_key=POSTHOG_API_KEY, host="https://us.posthog.com")
else:
    posthog = None

# Crear ID único por sesión
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Evento: App abierta
if posthog:
    posthog.capture(distinct_id=st.session_state.user_id,
        event="app_opened",
        properties={})
# ===============================
# HEADER PROFESIONAL INSTITUCIONAL
# ===============================

import datetime

st.markdown(
    """
    <style>
    .main-header {
        background-color: #f8f9fa;
        padding: 20px;
        border-bottom: 4px solid #003366;
        border-radius: 8px;
    }
    .title-software {
        font-size: 34px;
        font-weight: bold;
        text-align: center;
        color: #003366;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.image("ug.png", width=160)

with col2:
    st.markdown(
        """
        <div class="main-header">
            <div class="title-software">
                ANÁLISIS DE TAMAÑO DE BURBUJAS
            </div>
            <div class="subtitle">
                Universidad de Guanajuato<br>
                Departamento de Ingeniería en Minas, Metalurgia y Geología
                <br><br>
                Universidad de Colima<br>
                Facultad de Ciencias Químicas
                <br><br>
                Versión 1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.image("ucol.png", width=160)

st.markdown("---")

# ===============================
# ETAPAS
# ===============================

etapa = st.sidebar.radio(
    "Seleccione etapa:",
    ["1. Calibración",
     "2. ROI",
     "3. Ajustes",
     "4. Previsualización",
     "5. Batch",
     "6. Resultados"]
)

# ===============================
# ETAPA 1 – CALIBRACIÓN
# ===============================

if etapa == "1. Calibración":

    uploaded_files = st.file_uploader(
        "Subir imágenes",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        # Evento: imágenes cargadas
        if posthog:
            posthog.capture(
                st.session_state.user_id,
                'image_uploaded',
                {'num_images': len(uploaded_files)}
            )

        images = []

        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            images.append(np.array(img))

        st.session_state.images = images
        image_np = images[0]
        st.session_state.original = image_np

        st.write(f"{len(images)} imágenes cargadas")

        import io
        from PIL import Image

        pil_image = Image.fromarray(image_np).convert("RGB")

        # Convertir a PNG en memoria
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image_fixed = Image.open(buffer)

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="red",
            background_image=pil_image_fixed,
            update_streamlit=True,
            height=pil_image_fixed.height,
            width=pil_image_fixed.width,
            drawing_mode="line",
            key="calibration_canvas"
        )

        known_distance = st.number_input(
            "Distancia conocida (mm)",
            min_value=0.001
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]

            if len(objects) > 0:

                line = objects[0]
                x1, y1 = line["x1"], line["y1"]
                x2, y2 = line["x2"], line["y2"]

                pixel_distance = np.sqrt(
                    (x2 - x1) ** 2 + (y2 - y1) ** 2
                )

                if known_distance > 0 and pixel_distance > 0:
                    mm_per_pixel = known_distance / pixel_distance
                    st.session_state.mm_per_pixel = mm_per_pixel
                    st.success(
                        f"Calibración: {mm_per_pixel:.6f} mm/pixel"
)
# ===============================
# ETAPA 2 – ROI
# ===============================

elif etapa == "2. ROI":

    if "original" not in st.session_state:
        st.warning("Primero cargue imagen en calibración.")
    else:

        import io
        from PIL import Image

        image_np = st.session_state.original

        pil_image = Image.fromarray(image_np).convert("RGB")

        # 🔥 Convertir a PNG en memoria (necesario para Streamlit Cloud)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image_fixed = Image.open(buffer)

        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_width=2,
            stroke_color="green",
            background_image=pil_image_fixed,
            update_streamlit=True,
            height=pil_image_fixed.height,
            width=pil_image_fixed.width,
            drawing_mode="rect",
            key="roi_canvas"
        )

        if canvas_result.json_data is not None:

            objects = canvas_result.json_data["objects"]

            if len(objects) > 0:

                rect = objects[0]
                x = int(rect["left"])
                y = int(rect["top"])
                w = int(rect["width"])
                h = int(rect["height"])

                roi = image_np[y:y+h, x:x+w]

                # 🔹 Guardar ROI
                st.session_state.roi = roi

                # 🔹 Guardar coordenadas para batch
                st.session_state.roi_coords = (x, y, x+w, y+h)

                st.image(roi, caption="ROI seleccionada")

# ===============================
# ETAPA 3 - AJUSTES
# ===============================

elif etapa == "3. Ajustes":

    if "roi" not in st.session_state:
        st.warning("Primero defina ROI.")
    else:

        roi = st.session_state.roi

        st.subheader("Ajustes de preprocesamiento")

        # 🔹 CLAHE
        clip_limit = st.slider("CLAHE Clip Limit", 1.0, 5.0, 3.0, 0.1)
        tile_size = st.slider("CLAHE Tile Grid Size", 4, 16, 8, 1)

        # 🔹 Threshold fijo (coherente con batch)
        thresh_val = st.slider("Umbral binarización", 0, 255, 120)

        # 🔹 Fill Holes
        use_fill = st.checkbox("Aplicar Fill Holes", value=True)

        # 🔹 Closing
        closing_iter = st.slider("Iteraciones Morph Close", 0, 5, 1)

        # =============================
        # 🔹 Guardar parámetros en session_state
        # =============================

        st.session_state.clip_limit = clip_limit
        st.session_state.tile_size = tile_size
        st.session_state.threshold_value = thresh_val
        st.session_state.use_fill = use_fill
        st.session_state.closing_iter = closing_iter

        # =============================
        # 🔹 PROCESAMIENTO
        # =============================

        # CLAHE
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )

        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Escala de grises
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # Threshold fijo
        _, binary = cv2.threshold(
            gray,
            thresh_val,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Fill holes
        if use_fill:
            binary = ndi.binary_fill_holes(binary)
            binary = binary.astype(np.uint8) * 255

        # Closing
        if closing_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=closing_iter
            )

        # =============================
        # 🔹 Guardar resultados finales
        # =============================

        st.session_state.adjusted = enhanced
        st.session_state.binary = binary

        # Mostrar resultados
        col1, col2 = st.columns(2)
        col1.image(enhanced, channels="BGR", caption="Imagen mejorada")
        col2.image(binary, channels="GRAY", caption="Binaria sólida")

        st.success("Ajustes actualizados dinámicamente.")
# ===============================
# ETAPA 4 - PREVISUALIZACIÓN
# ===============================

elif etapa == "4. Previsualización":

    if "binary" not in st.session_state:
        st.warning("Primero realice los ajustes.")
    else:

        from skimage.segmentation import watershed
        from skimage.measure import label
        from skimage.morphology import h_maxima
        from skimage.feature import peak_local_max

        binary = st.session_state.binary.copy()
        original = st.session_state.roi.copy()

        st.image(binary, channels="GRAY", caption="Binaria para segmentación")

        st.subheader("Parámetros de filtrado")

        area_min = st.slider("Área mínima", 10, 5000, 50)
        area_max = st.slider("Área máxima", 100, 20000, 5000)
        circularidad_min = st.slider("Circularidad mínima", 0.0, 1.0, 0.4)

        st.subheader("Separación de burbujas tocándose")

        peak_factor = st.slider(
            "Sensibilidad separación",
            0.1, 0.6, 0.35, 0.01
        )

        st.session_state.area_min = area_min
        st.session_state.area_max = area_max
        st.session_state.circularidad_min = circularidad_min
        st.session_state.peak_factor = peak_factor

        if st.button("Previsualizar segmentación"):
            # Evento: análisis ejecutado
            if posthog:
                posthog.capture(st.session_state.user_id, 'analysis_executed')
            # =============================
            # 🔹 DISTANCE TRANSFORM ROBUSTO
            # =============================

            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)

            # =============================
            # 🔹 H-MAXIMA
            # =============================

            h_value = peak_factor * dist_smooth.max()
            hmax = h_maxima(dist_smooth, h_value)

            # =============================
            # 🔹 PEAK LOCAL MAX
            # =============================

            coordinates = peak_local_max(
                dist_smooth,
                min_distance=5,
                labels=binary,
                footprint=np.ones((3, 3))
            )

            markers = np.zeros(dist.shape, dtype=int)

            for i, coord in enumerate(coordinates):
                markers[coord[0], coord[1]] = i + 1

            markers = label(hmax | (markers > 0))

            # =============================
            # 🔹 WATERSHED
            # =============================

            labels = watershed(
                -dist_smooth,
                markers,
                mask=binary
            )

            output = original.copy()
            diameters = []

            for region_label in np.unique(labels):

                if region_label == 0:
                    continue

                mask_obj = np.zeros(binary.shape, dtype=np.uint8)
                mask_obj[labels == region_label] = 255

                cnts, _ = cv2.findContours(
                    mask_obj,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for c in cnts:

                    area = cv2.contourArea(c)
                    if area < area_min or area > area_max:
                        continue

                    perimeter = cv2.arcLength(c, True)
                    if perimeter == 0:
                        continue

                    circularity = 4 * np.pi * area / (perimeter**2)
                    if circularity < circularidad_min:
                        continue

                    (x, y), radius = cv2.minEnclosingCircle(c)

                    cv2.circle(
                        output,
                        (int(x), int(y)),
                        int(radius),
                        (0, 255, 0),
                        2
                    )

                    diameters.append(radius * 2)

            st.image(
                output,
                channels="BGR",
                caption="Previsualización segmentación"
            )

            st.session_state.preview_diameters = diameters
# ===============================
# ETAPA 5 - BATCH
# ===============================

elif etapa == "5. Batch":

    if "images" not in st.session_state:
        st.warning("No hay imágenes cargadas.")
    elif "area_min" not in st.session_state:
        st.warning("Primero ejecute la Previsualización.")
    else:

        from skimage.segmentation import watershed
        from skimage.measure import label
        from skimage.morphology import h_maxima
        from skimage.feature import peak_local_max

        if st.button("Procesar todas las imágenes"):

            all_results = []
            all_diameters = []

            for img in st.session_state.images:

                x1, y1, x2, y2 = st.session_state.roi_coords
                roi = img[y1:y2, x1:x2]

                # =============================
                # 🔹 PREPROCESAMIENTO
                # =============================

                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                clahe = cv2.createCLAHE(
                    clipLimit=st.session_state.clip_limit,
                    tileGridSize=(
                        st.session_state.tile_size,
                        st.session_state.tile_size
                    )
                )

                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

                _, binary_batch = cv2.threshold(
                    gray,
                    st.session_state.threshold_value,
                    255,
                    cv2.THRESH_BINARY_INV
                )

                if st.session_state.use_fill:
                    binary_batch = ndi.binary_fill_holes(binary_batch)
                    binary_batch = binary_batch.astype(np.uint8) * 255

                if st.session_state.closing_iter > 0:
                    kernel = np.ones((3,3), np.uint8)
                    binary_batch = cv2.morphologyEx(
                        binary_batch,
                        cv2.MORPH_CLOSE,
                        kernel,
                        iterations=st.session_state.closing_iter
                    )

                # =============================
                # 🔹 DISTANCE TRANSFORM ROBUSTO
                # =============================

                dist = cv2.distanceTransform(binary_batch, cv2.DIST_L2, 5)
                dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)

                # =============================
                # 🔹 H-MAXIMA
                # =============================

                h_value = (
                    st.session_state.peak_factor *
                    dist_smooth.max()
                )

                hmax = h_maxima(dist_smooth, h_value)

                # =============================
                # 🔹 PEAK LOCAL MAX
                # =============================

                coordinates = peak_local_max(
                    dist_smooth,
                    min_distance=5,
                    labels=binary_batch,
                    footprint=np.ones((3, 3))
                )

                markers = np.zeros(dist.shape, dtype=int)

                for i, coord in enumerate(coordinates):
                    markers[coord[0], coord[1]] = i + 1

                markers = label(hmax | (markers > 0))

                # =============================
                # 🔹 WATERSHED
                # =============================

                labels = watershed(
                    -dist_smooth,
                    markers,
                    mask=binary_batch
                )

                output = roi.copy()
                diameters = []

                for region_label in np.unique(labels):

                    if region_label == 0:
                        continue

                    mask_obj = np.zeros(
                        binary_batch.shape,
                        dtype=np.uint8
                    )

                    mask_obj[labels == region_label] = 255

                    cnts, _ = cv2.findContours(
                        mask_obj,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    for c in cnts:

                        area = cv2.contourArea(c)

                        if area < st.session_state.area_min or \
                           area > st.session_state.area_max:
                            continue

                        perimeter = cv2.arcLength(c, True)

                        if perimeter == 0:
                            continue

                        circularity = (
                            4 * np.pi * area /
                            (perimeter**2)
                        )

                        if circularity < \
                           st.session_state.circularidad_min:
                            continue

                        (x, y), radius = cv2.minEnclosingCircle(c)

                        cv2.circle(
                            output,
                            (int(x), int(y)),
                            int(radius),
                            (0, 255, 0),
                            2
                        )

                        diameters.append(radius * 2)

                all_results.append((output, diameters))
                all_diameters.extend(diameters)

            st.session_state.batch_results = all_results
            st.session_state.diameters = all_diameters

            st.success(
                f"Batch completado. "
                f"{len(all_diameters)} burbujas detectadas."
            )

        if "batch_results" in st.session_state:

            idx = st.slider(
                "Ver imagen",
                0,
                len(st.session_state.batch_results) - 1,
                0
            )

            st.image(
                st.session_state.batch_results[idx][0],
                channels="BGR"
            )

            st.write(
                f"Burbujas detectadas: "
                f"{len(st.session_state.batch_results[idx][1])}"
            )
# ===============================
# ETAPA 6 - RESULTADOS
# ===============================

elif etapa == "6. Resultados":

    import pandas as pd
    from io import BytesIO

    if "diameters" not in st.session_state:
        st.warning("Primero ejecute el batch.")
    else:

        diameters = np.array(st.session_state.diameters)

        if len(diameters) == 0:
            st.warning("No hay burbujas detectadas.")
        else:

            # 🔹 Conversión a mm
            if "mm_per_pixel" in st.session_state:
                diameters = diameters * st.session_state.mm_per_pixel

            # =============================
            # 🔹 TABLA BASE D32
            # =============================

            df_base = pd.DataFrame({
                "Diametro_mm": diameters,
                "Diametro^2": diameters**2,
                "Diametro^3": diameters**3
            })

            sum_d2 = df_base["Diametro^2"].sum()
            sum_d3 = df_base["Diametro^3"].sum()

            d32 = sum_d3 / sum_d2

            d10 = np.percentile(diameters, 10)
            d50 = np.percentile(diameters, 50)
            d90 = np.percentile(diameters, 90)

            st.subheader("Estadísticos")

            st.write(f"D10: {d10:.4f} mm")
            st.write(f"D50: {d50:.4f} mm")
            st.write(f"D90: {d90:.4f} mm")
            st.write(f"D32: {d32:.4f} mm")

            # =============================
            # 🔹 HISTOGRAMA
            # =============================

            counts, bins = np.histogram(diameters, bins=20)
            counts_percent = counts / counts.sum() * 100

            fig1, ax1 = plt.subplots()

            ax1.bar(
                bins[:-1],
                counts_percent,
                width=np.diff(bins),
                align="edge"
            )

            ax1.set_xlabel("Tamaño de burbuja (mm)")
            ax1.set_ylabel("Frecuencia (%)")
            ax1.set_title("Distribución de tamaño")

            st.pyplot(fig1)

            df_hist = pd.DataFrame({
                "Tamaño_mm": bins[:-1],
                "Frecuencia_%": counts_percent
            })

            # =============================
            # 🔹 CURVA ACUMULADA
            # =============================

            sorted_d = np.sort(diameters)
            cumulative = np.arange(1, len(sorted_d)+1) / len(sorted_d) * 100

            fig2, ax2 = plt.subplots()
            ax2.plot(sorted_d, cumulative)

            ax2.set_xlabel("Tamaño de burbuja (mm)")
            ax2.set_ylabel("Acumulado pasante (%)")
            ax2.set_title("Curva acumulada")

            st.pyplot(fig2)

            df_cum = pd.DataFrame({
                "Tamaño_mm": sorted_d,
                "Acumulado_%": cumulative
            })

            # =============================
            # 🔹 RESUMEN
            # =============================

            summary_df = pd.DataFrame({
                "Parametro": ["D10", "D50", "D90", "D32", "Sum_d2", "Sum_d3"],
                "Valor": [d10, d50, d90, d32, sum_d2, sum_d3]
            })

            # =============================
            # 🔹 EXPORTACIÓN A EXCEL
            # =============================

            output = BytesIO()

            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_base.to_excel(writer, sheet_name="Diametros_D32", index=False)
                df_hist.to_excel(writer, sheet_name="Histograma_XY", index=False)
                df_cum.to_excel(writer, sheet_name="Curva_Acumulada_XY", index=False)
                summary_df.to_excel(writer, sheet_name="Resumen", index=False)

            output.seek(0)

            st.download_button(
                label="Descargar Excel Completo",
                data=output,
                file_name="resultados_burbujas_completo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
# ===============================
# FOOTER INSTITUCIONAL
# ===============================

st.markdown(
    f"""
    <div class="footer">
        Desarrollado por Dr. Mario Alberto Corona Arroyo y Dr. Valentín Ibarra Galvan<br>
        © {datetime.datetime.now().year} Universidad de Guanajuato – Universidad de Colima
    </div>
    """,
    unsafe_allow_html=True

)










