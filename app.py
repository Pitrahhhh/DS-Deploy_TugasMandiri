import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore

# ---------- Konfigurasi halaman ----------
st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")

IMG_SIZE = (224, 224)
MODEL_PATH = "Model_MobileNetV2.keras"

# Disesuaikan persis dengan class_indices: {'CitraAI': 0, 'CitraAsli': 1}
CLASS_NAMES = ["CitraAI", "CitraAsli"]


@st.cache_resource
def load_classifier():
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🖼️ AI Generated vs Real Image Detector")
    st.write(
        "Unggah sebuah gambar untuk mengetahui apakah gambar tersebut "
        "**AI-generated** atau **real (asli)**."
    )

    model = load_classifier()

    uploaded_file = st.file_uploader(
        "Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        if st.button("🔍 Prediksi"):
            with st.spinner("Menganalisis gambar..."):
                processed = preprocess_image(image)
                prob = float(model.predict(processed)[0][0])

            # prob > 0.5 -> Indeks 1 (CitraAsli)
            # prob <= 0.5 -> Indeks 0 (CitraAI)
            if prob > 0.5:
                pred_index = 1
                confidence = prob
            else:
                pred_index = 0
                confidence = 1 - prob

            pred_class = CLASS_NAMES[pred_index]

            st.subheader("Hasil Deteksi")
            st.write(f"**Prediksi:** {pred_class}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            st.progress(confidence)

            # Tampilan output sesuai kelas
            if pred_class == "CitraAI":
                st.warning("Gambar ini terindikasi dibuat oleh AI (misal Midjourney, DALL-E, Stable Diffusion).")
            else:
                st.success("Gambar ini terindikasi sebagai foto asli.")


if __name__ == "__main__":
    main()
