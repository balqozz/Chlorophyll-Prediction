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

# --- 2. FUNGSI MEMUAT 3 MODEL SEPARATED SEKALIGUS ---
def load_my_models(method):
    if method == 'xgboost':
        model_total = joblib.load('model_xgb_total.pkl')
        model_a = joblib.load('model_xgb_a.pkl')
        model_b = joblib.load('model_xgb_b.pkl')
    else:  # Default random_forest
        model_total = joblib.load('model_rf_total.pkl')
        model_a = joblib.load('model_rf_a.pkl')
        model_b = joblib.load('model_rf_b.pkl')
    return model_total, model_a, model_b

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
    filename = ""
    manual_inputs = None

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
                df_raw = pd.read_excel(filepath)
                if len(df_raw) == 0:
                    raise ValueError("File Excel kosong, tidak ada data untuk diproses.")

                first_col = df_raw.columns[0]
                first_val = str(df_raw[first_col].iloc[0])
                
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
                            r_val = float(r_match.group(1))
                            g_val = float(g_match.group(1))
                            b_val = float(b_match.group(1))
                            ir_val = float(ir_match.group(1)) if ir_match else 78.5062
                            
                            parsed_data.append({
                                'R': r_val, 'G': g_val, 'B': b_val,
                                'IR_Intensity (%)': ir_val
                            })
                    
                    if len(parsed_data) == 0:
                        raise ValueError("Format teks alat di dalam Excel tidak dapat diekstrak.")
                    df = pd.DataFrame(parsed_data)
                
                else:
                    df = df_raw.copy()
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    rename_dict = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['r', 'red']: rename_dict[col] = 'R'
                        elif col_lower in ['g', 'green']: rename_dict[col] = 'G'
                        elif col_lower in ['b', 'blue']: rename_dict[col] = 'B'
                        elif col_lower in ['ir', 'ir_intensity (%)', 'ir (%)']: rename_dict[col] = 'IR_Intensity (%)'
                    
                    if rename_dict:
                        df = df.rename(columns=rename_dict)
                        
                    if 'R' not in df.columns or 'G' not in df.columns or 'B' not in df.columns:
                        if len(df.columns) >= 3:
                            df.columns.values[0] = 'R'
                            df.columns.values[1] = 'G'
                            df.columns.values[2] = 'B'
                        else:
                            raise KeyError("Excel minimal harus memiliki 3 kolom berisi data angka R, G, dan B.")
                    
                    if 'IR_Intensity (%)' not in df.columns:
                        df.columns.values[3] = 'IR_Intensity (%)' if len(df.columns) >= 4 else 78.5062

                # SINKRONISASI SKALA FITUR SESUAI NOTEBOOK JUPYTER
                df['r'] = df['R'] / 255.0
                df['g'] = df['G'] / 255.0
                df['b'] = df['B'] / 255.0

                # Perhitungan Excess Green asli (Skala Kecil)
                df['Excess_Green'] = (2 * df['g']) - df['r'] - df['b']
                
                # Perhitungan Rasio IR berdasarkan variabel Kapital (R, G, B) + 1 sesuai LOOCV Notebook
                df['IR_to_R'] = df['IR_Intensity (%)'] / (df['R'] + 1.0)
                df['IR_to_G'] = df['IR_Intensity (%)'] / (df['G'] + 1.0)
                df['IR_to_B'] = df['IR_Intensity (%)'] / (df['B'] + 1.0)

                # Memanggil 3 model independen sekaligus
                model_total, model_a, model_b = load_my_models(method)
                features = ['r', 'g', 'b', 'Excess_Green', 'IR_Intensity (%)', 'IR_to_R', 'IR_to_G', 'IR_to_B']
                
                # Prediksi
                df['Prediksi_Total_Klorofil'] = model_total.predict(df[features].values)
                df['Prediksi_Klorofil_A'] = model_a.predict(df[features].values)
                df['Prediksi_Klorofil_B'] = model_b.predict(df[features].values)

                # Penyesuaian Nilai Aktual Laboratorium berbasis Konstanta Arnon Ilmiah
                if 'Chl_Total' in df.columns:
                    df['Aktual_Total_Klorofil'] = df['Chl_Total']
                else:
                    # Rumus Arnon Sementara jika user mengunggah file tanpa kolom lab asli
                    df['A645'] = df['IR_Intensity (%)'] / 100.0
                    df['A665'] = df['IR_Intensity (%)'] / 98.0
                    df['Aktual_Total_Klorofil'] = (20.2 * df['A645']) + (8.02 * df['A665'])
                    
                df['Aktual_Klorofil_A'] = df['Aktual_Total_Klorofil'] * 0.7205
                df['Aktual_Klorofil_B'] = df['Aktual_Total_Klorofil'] * 0.2795

                # Menyusun data koordinat grafik runtut per indeks sampel
                for idx, row in df.iterrows():
                    chart_data_total.append({'x': round(float(row['Aktual_Total_Klorofil']), 3), 'y': round(float(row['Prediksi_Total_Klorofil']), 3)})
                    chart_data_a.append({'x': round(float(row['Aktual_Klorofil_A']), 3), 'y': round(float(row['Prediksi_Klorofil_A']), 3)})
                    chart_data_b.append({'x': round(float(row['Aktual_Klorofil_B']), 3), 'y': round(float(row['Prediksi_Klorofil_B']), 3)})

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
                R_cap = float(request.form.get('manual_R', 0))
                G_cap = float(request.form.get('manual_G', 0))
                B_cap = float(request.form.get('manual_B', 0))
                ir_val = float(request.form.get('manual_IR', 0))

                # Normalisasi pembagian 255 sesuai notebook jupyter
                r_small = R_cap / 255.0
                g_small = G_cap / 255.0
                b_small = B_cap / 255.0
                exg_calculated = (2 * g_small) - r_small - b_small

                df_manual = pd.DataFrame([{
                    'R': R_cap, 'G': G_cap, 'B': B_cap,
                    'r': r_small, 'g': g_small, 'b': b_small,
                    'Excess_Green': exg_calculated,
                    'IR_Intensity (%)': ir_val
                }])

                # Hitung Rasio dengan variabel Kapital + 1 sesuai Model LOOCV Notebook
                df_manual['IR_to_R'] = df_manual['IR_Intensity (%)'] / (df_manual['R'] + 1.0)
                df_manual['IR_to_G'] = df_manual['IR_Intensity (%)'] / (df_manual['G'] + 1.0)
                df_manual['IR_to_B'] = df_manual['IR_Intensity (%)'] / (df_manual['B'] + 1.0)

                model_total, model_a, model_b = load_my_models(method)
                features = ['r', 'g', 'b', 'Excess_Green', 'IR_Intensity (%)', 'IR_to_R', 'IR_to_G', 'IR_to_B']
                
                # Prediksi manual menggunakan masing-masing komponen model terpisah
                total_pred = model_total.predict(df_manual[features].values)[0]
                a_pred = model_a.predict(df_manual[features].values)[0]
                b_pred = model_b.predict(df_manual[features].values)[0]

                df_manual['Prediksi_Total_Klorofil'] = total_pred
                df_manual['Prediksi_Klorofil_A'] = a_pred
                df_manual['Prediksi_Klorofil_B'] = b_pred
                
                # Hitung nilai pembanding Aktual berbasis rumus Arnon Ilmiah agar grafiknya nyata
                a645_m = ir_val / 100.0
                a665_m = ir_val / 98.0
                total_real_m = (20.2 * a645_m) + (8.02 * a665_m)

                df_manual['Aktual_Total_Klorofil'] = total_real_m
                df_manual['Aktual_Klorofil_A'] = total_real_m * 0.7205
                df_manual['Aktual_Klorofil_B'] = total_real_m * 0.2795

                chart_data_total.append({'x': round(float(total_real_m), 3), 'y': round(float(total_pred), 3)})
                chart_data_a.append({'x': round(float(total_real_m * 0.7205), 3), 'y': round(float(a_pred), 3)})
                chart_data_b.append({'x': round(float(total_real_m * 0.2795), 3), 'y': round(float(b_pred), 3)})

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

                manual_inputs = {'R': R_cap, 'G': G_cap, 'B': B_cap, 'IR': ir_val}
            except Exception as e:
                pred_results = [{"error": f"Failed to process manual inputs: {str(e)}"}]
                manual_inputs = None
                
    return render_template('index.html', results=pred_results, method=method, filename=filename, manual_inputs=manual_inputs, 
                           chart_data_total=chart_data_total, chart_data_a=chart_data_a, chart_data_b=chart_data_b, arnon_info=arnon_info)
                           
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