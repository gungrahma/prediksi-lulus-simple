from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

# Memuat Model AI yang sudah dibuat di train.py
model_ai = joblib.load("model_kelulusan.joblib")

class DataSiswa(BaseModel):
    nama: str
    # Update : Jam belajar ada restriction sampai 24 jam, lebih dari 24 = error sesuai dengan human.
    jam_belajar: int = Field(ge=0, le=24, description="Jam belajar dalam sehari")
    ikut_bimbel: bool

@app.post("/predict")
def prediksi(data : DataSiswa):
    # Persiapan data sesuai dengan scikit
    # Scikit butuh angka, Bool awalnya False/True jadi angka 1/0
    status_bimbel = 1 if data.ikut_bimbel else 0

    fitur_input = [[data.jam_belajar, status_bimbel]]

    hasil_prediksi = model_ai.predict(fitur_input)

    status = "Lulus" if hasil_prediksi[0] == 1 else "Tidak Lulus"

    return {
        "Pesan": f"Analisis AI untuk {data.nama} selesai!",
        "input_jam_belajar": data.jam_belajar,
        "status_ikut_bimbel": "Ikut" if data.ikut_bimbel else "Tidak ikut",
        "prediksi_status": status
    }