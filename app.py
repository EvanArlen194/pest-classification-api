# Import library yang dibutuhkan
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import json
from PIL import Image, UnidentifiedImageError
import io

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Konfigurasi CORS untuk mengizinkan akses dari semua origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Mengizinkan semua domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path ke model dan label class
MODEL_PATH = 'saved_model/model_hama.h5'
LABELS_PATH = 'saved_model/class_labels.json'

# Load model yang sudah dilatih
model = load_model(MODEL_PATH)

# Load label class dari file JSON
with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)


# Daftar saran penanganan berdasarkan jenis hama
suggestions = {
    "Semut": "Taburkan kapur semut di sekitar tanaman atau semprot dengan air yang dicampur sedikit sabun batang yang biasa dipakai cuci tangan. Jangan pakai sabun cuci piring atau deterjen karena bisa merusak tanaman.",
    "Lebah": "Jika lebah tidak mengganggu, biarkan saja karena mereka membantu penyerbukan. Jika sarangnya dekat tanaman dan berbahaya, minta bantuan petugas pengendali hama.",
    "Kumbang": "Semprot bagian tanaman yang banyak kumbangnya dengan minyak dari daun mimba atau air sabun seperti tadi. Jika sedikit, ambil kumbangnya satu per satu.",
    "Ulat": "Petik ulat yang terlihat di tanaman. Bisa juga disemprot dengan air yang dicampur minyak daun mimba.",
    "Cacing tanah": "Cacing tanah baik untuk tanah dan tanaman. Tidak perlu diatasi kecuali jumlahnya terlalu banyak.",
    "Earwig": "Pasang jebakan dengan piring berisi minyak goreng di malam hari. Bersihkan juga daun mati dan sampah di sekitar tanaman.",
    "Belalang": "Pasang jaring pelindung di tanaman atau semprot dengan air yang dicampur cabai dan bawang putih yang sudah dihaluskan untuk mengusir.",
    "Ngengat": "Pasang lampu perangkap di malam hari supaya ngengat tertarik. Bisa juga semprot dengan minyak dari daun mimba.",
    "Siput Tanpa Cangkang": "Bersihkan rumput liar dan gulma (tanaman pengganggu) di sekitar tanaman. Gunakan umpan keong yang aman (iron phosphate) dari toko pertanian.",
    "Siput Bercangkang": "Taburkan kulit telur yang sudah dihancurkan di sekitar tanaman atau buat jebakan dengan air gula dan ragi: campur air hangat, gula, dan sedikit ragi dalam wadah dangkal, letakkan dekat tanaman. Siput akan tertarik dan masuk ke jebakan.",
    "Tawon": "Jangan ganggu sarangnya supaya tidak marah. Bisa disemprot dengan air yang dicampur sedikit sabun batang yang biasa dipakai cuci tangan. Jangan pakai sabun cuci piring atau deterjen karena bisa merusak tanaman. Kalau bahaya, minta bantuan petugas pengendali hama.",
    "Kutu Beras": "Simpan hasil panen di tempat tertutup rapat. Bisa taruh daun salam atau serai untuk mengusir kutu secara alami."
}

# Batas maksimal ukuran file yang diizinkan (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Format file yang diizinkan untuk diunggah
allowed_extensions = {"jpg", "jpeg", "png"}

# Endpoint utama untuk prediksi gambar hama
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validasi nama file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty file name")

    # Validasi ekstensi file
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Format file tidak didukung. Gunakan JPG atau PNG.")

    # Baca isi file dan cek ukuran
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran file terlalu besar. Maksimal 5MB.")

    # Coba konversi ke gambar valid
    try:
        img = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid")

    # Proses gambar: ubah ke RGB, ubah ukuran, normalisasi, dan bentuk array untuk prediksi
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi menggunakan model
    try:
        prediction = model.predict(img_array)
    except Exception:
        raise HTTPException(status_code=500, detail="Model gagal melakukan prediksi. Periksa format gambar.")

    # Hitung kepercayaan (confidence) prediksi
    confidence = float(np.max(prediction))
    confidence_percent = round(confidence * 100, 2)

    # Jika confidence terlalu rendah, anggap gambar tidak relevan
    if confidence < 0.93:
        raise HTTPException(
            status_code=400,
            detail="Gambar yang diunggah tampaknya tidak menunjukkan keberadaan hama tanaman seperti serangga. Sistem tidak dapat melakukan identifikasi hama berdasarkan gambar ini."
        )

    # Ambil indeks dan nama class dari prediksi
    class_index = int(np.argmax(prediction))
    class_name_id = class_labels[class_index]

     # Ambil saran berdasarkan jenis hama yang dikenali
    suggestion = suggestions.get(class_name_id, "Belum ada saran yang tersedia untuk saat ini.")

    # Kembalikan hasil prediksi dan saran dalam format JSON
    return {
        "data": {
            "prediction": class_name_id,
            "confidence": f"{confidence_percent}%",
            "suggestion": suggestion
        }
    }