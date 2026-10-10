import pandas as pd
import numpy as np
import os
import traceback

# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_dir = r'c:\Users\faruk\PycharmProjects\TübitakDeneme\T-bitakVerileri\ŞirketVerileri'
companies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

macro_data = {}
try:
    print("Tüm EVDS Excel dosyalarından (Dolar, Enflasyon Oranı, TÜFE Düzey) veriler çekiliyor...")
    usd_path = r'c:\Users\faruk\PycharmProjects\TübitakDeneme\T-bitakVerileri\EnflasyonVeDolar\EVDS_10-10-2026_Dolar.xlsx'
    enf_path = r'c:\Users\faruk\PycharmProjects\TübitakDeneme\T-bitakVerileri\EnflasyonVeDolar\EVDS_10-10-2026_Enflasyon_Yıllık.xlsx'
    cpi_path = r'c:\Users\faruk\PycharmProjects\TübitakDeneme\T-bitakVerileri\EnflasyonVeDolar\EVDS_10-10-2026_Enflasyon_Düzey.xlsx'
    
    df_usd = pd.read_excel(usd_path)
    df_enf = pd.read_excel(enf_path)
    df_cpi = pd.read_excel(cpi_path)
    
    usd_col = [c for c in df_usd.columns if 'TP_DK_USD' in c or 'USD' in c][0]
    enf_col = [c for c in df_enf.columns if 'TP_TUKFIY' in c or 'TUK' in c][0]
    cpi_col = [c for c in df_cpi.columns if 'TP_TUKFIY' in c or 'TUK' in c][0]
    
    for _, row in df_usd.iterrows():
        key = str(row['Tarih']).strip()
        if key not in macro_data: macro_data[key] = {}
        macro_data[key]['usd'] = float(row[usd_col])
            
    for _, row in df_enf.iterrows():
        key = str(row['Tarih']).strip()
        if key not in macro_data: macro_data[key] = {}
        macro_data[key]['enf'] = float(row[enf_col])
            
    for _, row in df_cpi.iterrows():
        key = str(row['Tarih']).strip()
        if key not in macro_data: macro_data[key] = {}
        macro_data[key]['cpi'] = float(row[cpi_col])
            
    print("Tüm EVDS Entegrasyonları Başarılı!")
except Exception as e:
    print(f"EVDS Entegrasyon Hatası: {e}")

df_yeni = pd.DataFrame(df_usd.iterrows())
print(df_yeni)