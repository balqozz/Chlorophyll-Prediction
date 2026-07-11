import os
import joblib
import pandas as pd
import numpy as np
import re
import sqlite3  # <-- Library SQLite
import json     # Untuk menyimpan data array grafik ke database dalam bentuk string teks
from flask import Flask, request, render_template, abort
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
DB_NAME = 'skripsi_online.db'  # <-- Nama file database kamu

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ==========================================================================
# CATATAN PERBAIKAN (supaya sinkron dengan notebook ekstraksi terbaru):
# - Model sekarang dilatih dengan 5 FITUR SAJA: R, G, B, IR, ExG (skala
#   MENTAH, bukan dinormalisasi 0-1, dan TANPA rasio IR_to_R/G/B). Fitur
#   lama (r,g,b normalisasi + Excess_Green skala kecil + IR_to_R/G/B) sudah
#   tidak dipakai lagi -- kalau tetap dipakai, jumlah fitur akan mismatch
#   dengan model_rf_a.pkl / model_rf_b.pkl / model_rf_total.pkl yang baru.
# - "Aktual" klorofil TIDAK LAGI direka dari rumus IR/97/98 + rasio
#   72,05%/27,95%. Nilai aktual yang benar hanya ada kalau file yang
#   diupload memang membawa kolom ground truth asli (Klorofil_A,
#   Klorofil_B, Total_Klorofil), hasil dari rumus Arnon berbasis
#   absorbansi lab (lihat notebook -> Ground_Truth_Absorbansi_Arnon.xlsx).
#   Untuk sampel baru yang belum pernah diukur di lab, sistem HANYA
#   menampilkan prediksi (tanpa nilai aktual palsu).
# ==========================================================================

FITUR = ['R', 'G', 'B', 'IR', 'ExG']

# ==========================================================================
# TABEL GROUND TRUTH PER LABEL (hasil rumus Arnon dari absorbansi UV-Vis asli)
# Dipakai untuk AUTO-LOOKUP nilai aktual saat file mentah HH01-HH34 diupload,
# karena file-file itu sebenarnya BUKAN "tanaman baru" -- mereka adalah 16
# tanaman yang sama yang dipakai untuk training, jadi label & ground truth-nya
# sudah diketahui. Nilai ini HARUS SAMA dengan Ground_Truth_Absorbansi_Arnon.xlsx.
# ==========================================================================
GROUND_TRUTH_PER_LABEL = {
    0: {'Klorofil_A': 6.186804, 'Klorofil_B': 11.979488, 'Total_Klorofil': 18.127974},
    1: {'Klorofil_A': 6.738629, 'Klorofil_B': 12.791618, 'Total_Klorofil': 19.494740},
    2: {'Klorofil_A': 6.739630, 'Klorofil_B': 12.793440, 'Total_Klorofil': 19.497562},
    3: {'Klorofil_A': 6.849696, 'Klorofil_B': 12.965544, 'Total_Klorofil': 19.777296},
}


def cari_label_dari_nama_file(filename):
    """Deteksi apakah file yang diupload adalah salah satu dari 16 tanaman
    training asli (format nama HHxy, x=label 0-3, y=tanaman ke-1..4).
    Kalau cocok, kembalikan label-nya (0-3). Kalau tidak, return None."""
    nama = os.path.splitext(os.path.basename(filename))[0].upper()
    m = re.match(r'^HH([0-3])([1-4])$', nama)
    if m:
        return int(m.group(1))
    return None


# --- 1. FUNGSI INISIALISASI DATABASE SQLite ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Membuat tabel history jika belum ada
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            method TEXT,
            timestamp TEXT,
            chart_data_total TEXT,
            chart_data_a TEXT,
            chart_data_b TEXT,
            results TEXT
        )
    ''')
    conn.commit()
    conn.close()


# Jalankan inisialisasi tabel saat web pertama kali di-run
init_db()


# --- 2. FUNGSI MEMUAT 3 MODEL RANDOM FOREST ---
# Catatan: opsi XGBoost sudah dihapus dari halaman web karena scope skripsi
# ini "exclusively Random Forest" (lihat Bab 1.3). Kalau nanti mau
# menambahkan XGBoost lagi sebagai perbandingan, XGBoost HARUS dilatih ulang
# dulu memakai 5 fitur [R,G,B,IR,ExG] yang sama seperti Random Forest,
# supaya konsisten dengan pipeline yang sekarang.
def load_my_models(method):
    model_total = joblib.load('model_rf_total.pkl')
    model_a = joblib.load('model_rf_a.pkl')
    model_b = joblib.load('model_rf_b.pkl')
    return model_total, model_a, model_b


def cari_baris_header(filepath, max_scan=10):
    """Beberapa file Excel (misal Hasil_Prediksi_RandomForest.xlsx) punya judul
    & catatan di baris-baris atas sebelum baris header sebenarnya (Nama_File, R,
    G, B, ...). Fungsi ini mencari baris mana yang benar-benar berisi nama
    kolom, supaya pd.read_excel tidak salah menganggap teks judul sebagai header."""
    df_scan = pd.read_excel(filepath, header=None, nrows=max_scan)
    for i in range(len(df_scan)):
        nilai_baris = [str(v).strip().upper() for v in df_scan.iloc[i].tolist()]
        if 'R' in nilai_baris and 'G' in nilai_baris and 'B' in nilai_baris:
            return i
    return 0  # kalau tidak ketemu, anggap saja baris pertama (perilaku lama)


def hitung_ExG(df):
    """ExG dihitung dari R,G,B skala MENTAH -- HARUS SAMA dengan cara
    training di notebook (ekstrak_satu_file -> ExG = 2G - R - B)."""
    df['ExG'] = (2 * df['G']) - df['R'] - df['B']
    return df


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    method = request.form.get('method') or request.args.get('method') or 'random_forest'
    pred_results = []
    chart_data_total = []
    chart_data_a = []
    chart_data_b = []
    chart_labels = []
    filename = ""
    manual_inputs = None
    ada_ground_truth = False   # True hanya kalau file upload membawa kolom Klorofil_A/B/Total asli

    arnon_info = {
        'rumus_a': 'Chl a = 12.7 * A663 - 2.69 * A645',
        'rumus_b': 'Chl b = 22.9 * A645 - 4.68 * A669',
        'rumus_total': 'Total Chl = 20.2 * A645 + 8.02 * A665'
    }

    if request.method == 'POST':
        file = request.files.get('file')
        input_type = request.form.get('input_type')

        # --- JALUR A: UPLOAD EXCEL ---
        if input_type == 'file' and file and file.filename != '':
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                baris_header = cari_baris_header(filepath)
                df_raw = pd.read_excel(filepath, header=baris_header)
                if len(df_raw) == 0:
                    raise ValueError("File Excel kosong, tidak ada data untuk diproses.")

                first_col = df_raw.columns[0]
                first_val = str(df_raw[first_col].iloc[0])

                # Kasus 1: file berisi teks mentah sensor "R= .. G= .. B= .. %"
                # (ini format 1 file = 1 tanaman, seperti HH01.xlsx - HH34.xlsx)
                if 'R=' in first_val or 'R=' in str(first_col):
                    parsed_data = []
                    all_lines = []
                    if 'R=' in str(first_col):
                        all_lines.append(str(first_col))

                    for val in df_raw[first_col].dropna():
                        all_lines.append(str(val))

                    for line in all_lines:
                        r_match = re.search(r'R=\s*([\d\.]+)', line)
                        g_match = re.search(r'G=\s*([\d\.]+)', line)
                        b_match = re.search(r'B=\s*([\d\.]+)', line)
                        ir_match = re.search(r'([\d\.]+)\s*%', line)

                        if r_match and g_match and b_match:
                            parsed_data.append({
                                'R': float(r_match.group(1)),
                                'G': float(g_match.group(1)),
                                'B': float(b_match.group(1)),
                                'IR': float(ir_match.group(1)) if ir_match else np.nan
                            })

                    if len(parsed_data) == 0:
                        raise ValueError("Format teks alat di dalam Excel tidak dapat diekstrak.")

                    df_semua_baris = pd.DataFrame(parsed_data)
                    # Hitung ExG per baris DULU (sama seperti saat training),
                    # baru dirata-ratakan jadi 1 baris yang mewakili 1 tanaman.
                    # Ini penting -- kalau tidak dirata-ratakan, hasilnya akan
                    # beda dengan cara model dilatih (model dilatih dari fitur
                    # rata-rata per tanaman, bukan per baris pembacaan sensor).
                    df_semua_baris = hitung_ExG(df_semua_baris)
                    rata_rata = df_semua_baris[['R', 'G', 'B', 'IR', 'ExG']].mean(numeric_only=True)
                    df = pd.DataFrame([rata_rata])

                    print(f"[INFO] {filename}: {len(df_semua_baris)} baris sensor dirata-ratakan "
                          f"jadi 1 baris fitur (sesuai cara training).")

                    # Auto-lookup ground truth kalau nama file cocok HH01-HH34
                    label_terdeteksi = cari_label_dari_nama_file(filename)
                    if label_terdeteksi is not None:
                        gt = GROUND_TRUTH_PER_LABEL[label_terdeteksi]
                        df['Perlakuan'] = label_terdeteksi
                        df['Klorofil_A'] = gt['Klorofil_A']
                        df['Klorofil_B'] = gt['Klorofil_B']
                        df['Total_Klorofil'] = gt['Total_Klorofil']
                        print(f"[INFO] {filename} terdeteksi sebagai tanaman label {label_terdeteksi} "
                              f"(salah satu dari 16 tanaman training) -> ground truth otomatis ditemukan.")

                # Kasus 2: file sudah berupa tabel kolom (R, G, B, IR, ...)
                else:
                    df = df_raw.copy()
                    df.columns = [str(c).strip() for c in df.columns]

                    rename_dict = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['r', 'red']: rename_dict[col] = 'R'
                        elif col_lower in ['g', 'green']: rename_dict[col] = 'G'
                        elif col_lower in ['b', 'blue']: rename_dict[col] = 'B'
                        elif col_lower in ['ir', 'ir_intensity (%)', 'ir (%)']: rename_dict[col] = 'IR'
                    if rename_dict:
                        df = df.rename(columns=rename_dict)

                    if 'R' not in df.columns or 'G' not in df.columns or 'B' not in df.columns:
                        raise KeyError("Excel harus memiliki kolom R, G, dan B (bisa juga IR).")
                    if 'IR' not in df.columns:
                        df['IR'] = np.nan

                    # Bonus: kalau tabel sudah punya kolom Perlakuan tapi belum
                    # ada target klorofil, isi otomatis dari tabel ground truth.
                    if 'Perlakuan' in df.columns and not all(
                            k in df.columns for k in ['Klorofil_A', 'Klorofil_B', 'Total_Klorofil']):
                        gt_df = df['Perlakuan'].astype(int).map(GROUND_TRUTH_PER_LABEL)
                        df['Klorofil_A'] = gt_df.map(lambda d: d['Klorofil_A'] if isinstance(d, dict) else np.nan)
                        df['Klorofil_B'] = gt_df.map(lambda d: d['Klorofil_B'] if isinstance(d, dict) else np.nan)
                        df['Total_Klorofil'] = gt_df.map(lambda d: d['Total_Klorofil'] if isinstance(d, dict) else np.nan)

                # Hitung ExG (skala mentah, SAMA seperti saat training)
                df = hitung_ExG(df)

                # Memanggil 3 model independen sekaligus
                model_total, model_a, model_b = load_my_models(method)
                df['Prediksi_Klorofil_A'] = model_a.predict(df[FITUR].values)
                df['Prediksi_Klorofil_B'] = model_b.predict(df[FITUR].values)
                df['Prediksi_Total_Klorofil'] = model_total.predict(df[FITUR].values)

                # Label sumbu-X untuk grafik per-sampel (2 warna): pakai
                # Nama_File kalau ada, kalau tidak pakai nomor urut sampel.
                if 'Nama_File' in df.columns:
                    chart_labels = df['Nama_File'].astype(str).tolist()
                else:
                    chart_labels = [filename if len(df) == 1 else f"Sampel {i+1}" for i in range(len(df))]

                # Ground truth HANYA dipakai kalau memang ada di file upload
                # atau berhasil ditemukan otomatis di atas
                if all(k in df.columns for k in ['Klorofil_A', 'Klorofil_B', 'Total_Klorofil']) and \
                        df[['Klorofil_A', 'Klorofil_B', 'Total_Klorofil']].notna().all().all():
                    ada_ground_truth = True
                    for idx, row in df.iterrows():
                        chart_data_a.append({'x': round(float(row['Klorofil_A']), 4),
                                              'y': round(float(row['Prediksi_Klorofil_A']), 4)})
                        chart_data_b.append({'x': round(float(row['Klorofil_B']), 4),
                                              'y': round(float(row['Prediksi_Klorofil_B']), 4)})
                        chart_data_total.append({'x': round(float(row['Total_Klorofil']), 4),
                                                  'y': round(float(row['Prediksi_Total_Klorofil']), 4)})

                pred_results = df.to_dict(orient='records')

                # --- JALUR SQLite: SIMPAN REKORD BARU ---
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO history (filename, method, timestamp, chart_data_total, chart_data_a, chart_data_b, results)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (filename, method, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      json.dumps(chart_data_total), json.dumps(chart_data_a), json.dumps(chart_data_b), json.dumps(pred_results)))
                conn.commit()
                conn.close()

            except Exception as e:
                pred_results = [{"error": f"Gagal memproses file Excel: {str(e)}"}]

        # --- JALUR B: MANUAL INPUT ---
        elif input_type == 'manual':
            filename = "Manual Input"
            try:
                R_val = float(request.form.get('manual_R', 0))
                G_val = float(request.form.get('manual_G', 0))
                B_val = float(request.form.get('manual_B', 0))
                IR_val = float(request.form.get('manual_IR', 0))
                ExG_val = (2 * G_val) - R_val - B_val   # sama seperti di notebook

                df_manual = pd.DataFrame([{'R': R_val, 'G': G_val, 'B': B_val,
                                            'IR': IR_val, 'ExG': ExG_val}])

                model_total, model_a, model_b = load_my_models(method)
                a_pred = model_a.predict(df_manual[FITUR].values)[0]
                b_pred = model_b.predict(df_manual[FITUR].values)[0]
                total_pred = model_total.predict(df_manual[FITUR].values)[0]

                df_manual['Prediksi_Klorofil_A'] = a_pred
                df_manual['Prediksi_Klorofil_B'] = b_pred
                df_manual['Prediksi_Total_Klorofil'] = total_pred

                # Tidak ada nilai aktual untuk input manual (daun belum diukur di lab)
                pred_results = df_manual.to_dict(orient='records')

                # --- JALUR SQLite: SIMPAN MANUAL INPUT ---
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO history (filename, method, timestamp, chart_data_total, chart_data_a, chart_data_b, results)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (filename, method, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      json.dumps(chart_data_total), json.dumps(chart_data_a), json.dumps(chart_data_b), json.dumps(pred_results)))
                conn.commit()
                conn.close()

                manual_inputs = {'R': R_val, 'G': G_val, 'B': B_val, 'IR': IR_val}
            except Exception as e:
                pred_results = [{"error": f"Failed to process manual inputs: {str(e)}"}]
                manual_inputs = None

    return render_template('index.html', results=pred_results, method=method, filename=filename,
                           manual_inputs=manual_inputs, chart_data_total=chart_data_total,
                           chart_data_a=chart_data_a, chart_data_b=chart_data_b,
                           chart_labels=chart_labels,
                           arnon_info=arnon_info, ada_ground_truth=ada_ground_truth)


# --- 3. JALUR SQLite: AMBIL SEMUA DAFTAR RIWAYAT ---
@app.route('/history')
def history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, method, timestamp FROM history ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    history_list = []
    for r in rows:
        history_list.append({
            'id': r[0],
            'filename': r[1],
            'method': r[2],
            'timestamp': r[3]
        })
    return render_template('history.html', history=history_list)


# --- 4. JALUR SQLite: AMBIL DETAIL ARSIP REPORT BERDASARKAN ID ---
@app.route('/history/<int:entry_id>')
def history_detail(entry_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, method, timestamp, chart_data_total, chart_data_a, chart_data_b, results FROM history WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        abort(404)

    detail_data = {
        'filename': row[0],
        'method': row[1],
        'timestamp': row[2],
        'chart_data_total': json.loads(row[3]),
        'chart_data_a': json.loads(row[4]),
        'chart_data_b': json.loads(row[5]),
        'results': json.loads(row[6])
    }
    return render_template('history_detail.html', history=detail_data, entry_id=entry_id)


if __name__ == '__main__':
    app.run(debug=True)