import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore

# ---------- Konfigurasi halaman ----------
st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")

IMG_SIZE = (224, 224)
MODEL_PATH = "Model_MobileNetV2.keras"
CLASS_NAMES = ["CitraAsli", "CitraAI"]

# Index kelas "CitraAI" di CLASS_NAMES di atas — WAJIB dicek ke train_gen.class_indices
# hasil training (dicetak otomatis di notebook). Ganti angkanya kalau urutannya beda.
AI_CLASS_INDEX = CLASS_NAMES.index("CitraAI")


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


            pred_index = 1 if prob > 0.5 else 0
            pred_class = CLASS_NAMES[pred_index]
            confidence = prob if prob > 0.5 else 1 - prob

            is_ai = pred_index == AI_CLASS_INDEX

            st.subheader("Hasil Deteksi")
            st.write(f"**Prediksi:** {pred_class}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            st.progress(confidence)

            if is_ai:
                st.warning("Gambar ini terindikasi dibuat oleh AI.")
            else:
                st.success("Gambar ini terindikasi sebagai foto asli.")


if __name__ == "__main__":
    main()
