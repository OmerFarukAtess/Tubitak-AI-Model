import pandas as pd
import numpy as np
import os
import traceback

# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_dir = r'c:\Users\faruk\PycharmProjects\TübitakDeneme\T-bitakVerileri\tübitak veriler'
companies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# TÜRKİYE EKONOMİSİ MAKRO VERİLERİ (2020-2024)
macro_data = {
    '2020/3':  {'cpi': 450,  'enf': 11.8, 'usd': 6.50},
    '2020/6':  {'cpi': 460,  'enf': 12.6, 'usd': 6.85},
    '2020/9':  {'cpi': 475,  'enf': 11.7, 'usd': 7.35},
    '2020/12': {'cpi': 500,  'enf': 14.6, 'usd': 7.80},
    
    '2021/3':  {'cpi': 525,  'enf': 16.1, 'usd': 7.50},
    '2021/6':  {'cpi': 550,  'enf': 17.5, 'usd': 8.60},
    '2021/9':  {'cpi': 580,  'enf': 19.5, 'usd': 8.50},
    '2021/12': {'cpi': 686,  'enf': 36.0, 'usd': 13.50},
    
    '2022/3':  {'cpi': 843,  'enf': 61.1, 'usd': 14.50},
    '2022/6':  {'cpi': 977,  'enf': 78.6, 'usd': 16.50},
    '2022/9':  {'cpi': 1046, 'enf': 83.4, 'usd': 18.20},
    '2022/12': {'cpi': 1128, 'enf': 64.2, 'usd': 18.60},
    
    '2023/3':  {'cpi': 1269, 'enf': 50.5, 'usd': 19.10},
    '2023/6':  {'cpi': 1351, 'enf': 38.2, 'usd': 25.80},
    '2023/9':  {'cpi': 1700, 'enf': 61.5, 'usd': 27.20},
    '2023/12': {'cpi': 1859, 'enf': 64.7, 'usd': 29.00},
    
    '2024/3':  {'cpi': 2110, 'enf': 68.5, 'usd': 31.50},
    '2024/6':  {'cpi': 2250, 'enf': 71.6, 'usd': 32.20},
    '2024/9':  {'cpi': 2350, 'enf': 49.3, 'usd': 34.00},
}
BASE_CPI = 450.0  # 2020 Mart baz yılı

def get_row(df, keywords):
    for idx in df.index:
        if type(idx) == str:
            if any(kw.lower() in idx.lower() for kw in keywords):
                res = df.loc[idx]
                if isinstance(res, pd.DataFrame):
                    return res.iloc[0]
                return res
    return None

data_list = []
print("2020'den 2024'e kadar tüm veriler birleştiriliyor... (Enflasyon Düzeltmeli)")

for company in companies:
    file_2020 = os.path.join(base_dir, company, f"{company}2020.xlsx")
    file_2021 = os.path.join(base_dir, company, f"{company}2021.xlsx")
    file_2022 = os.path.join(base_dir, company, f"{company}2022.xlsx")
    file_2023 = os.path.join(base_dir, company, f"{company}2023.xlsx")
    file_2024 = os.path.join(base_dir, company, f"{company}2024.xlsx")
    
    if not (os.path.exists(file_2020) and os.path.exists(file_2021) and os.path.exists(file_2022) and os.path.exists(file_2023) and os.path.exists(file_2024)):
        continue
        
    try:
        df_2020 = pd.read_excel(file_2020, index_col=0)
        df_2021 = pd.read_excel(file_2021, index_col=0)
        df_2022 = pd.read_excel(file_2022, index_col=0)
        df_2023 = pd.read_excel(file_2023, index_col=0)
        df_2024 = pd.read_excel(file_2024, index_col=0)
        
        cols_2020 = sorted([c for c in df_2020.columns if type(c) == str and '2020/' in c], key=lambda x: int(x.split('/')[1]))
        cols_2021 = sorted([c for c in df_2021.columns if type(c) == str and '2021/' in c], key=lambda x: int(x.split('/')[1]))
        cols_2022 = sorted([c for c in df_2022.columns if type(c) == str and '2022/' in c], key=lambda x: int(x.split('/')[1]))
        cols_2023 = sorted([c for c in df_2023.columns if type(c) == str and '2023/' in c], key=lambda x: int(x.split('/')[1]))
        cols_2024 = sorted([c for c in df_2024.columns if type(c) == str and '2024/' in c], key=lambda x: int(x.split('/')[1]))
        
        df_combined = pd.concat([df_2020[cols_2020], df_2021[cols_2021], df_2022[cols_2022], df_2023[cols_2023], df_2024[cols_2024]], axis=1)
        
        tv = get_row(df_combined, ["TOPLAM VARLIKLAR"])
        ozk = get_row(df_combined, ["zkaynaklar", "Özkaynaklar"])
        nk = get_row(df_combined, ["NEM KARI (ZARARI)", "DÖNEM KARI"])
        donen_varlik = get_row(df_combined, ["Dönen Varlıklar", "Dnen Varlklar"])
        kisa_v_yukum = get_row(df_combined, ["Kısa Vadeli Yükümlülükler", "Ksa Vadeli"])
        satis = get_row(df_combined, ["Satış Gelirleri", "Satş Gelirleri"])
        
        nakit = get_row(df_combined, ["Nakit ve Nakit Benzerleri", "Nakit"])
        brut_kar = get_row(df_combined, ["BRÜT KAR", "BRT KAR"])
        faal_kar = get_row(df_combined, ["FAALİYET KARI", "FAALYET KARI", "Faaliyet Karı"])
        uzun_v_yukum = get_row(df_combined, ["Uzun Vadeli Yükümlülükler", "Uzun Vadeli"])
        stoklar = get_row(df_combined, ["Stoklar"])
        maliyet = get_row(df_combined, ["Satışların Maliyeti", "Satşlarn Maliyeti"])
        
        if tv is None or nk is None or ozk is None:
            continue
            
        all_cols = cols_2020 + cols_2021 + cols_2022 + cols_2023 + cols_2024
        
        for i in range(len(all_cols) - 1):
            current_col = all_cols[i]
            next_col = all_cols[i+1]
            
            cur_macro = macro_data.get(current_col)
            next_macro = macro_data.get(next_col)
            
            if not cur_macro or not next_macro:
                continue
                
            deflator = BASE_CPI / cur_macro['cpi'] 
            next_deflator = BASE_CPI / next_macro['cpi']
            
            try:
                # Girdi Değerleri - ENFLASYON DÜZELTMELİ
                val_tv = float(tv[current_col]) * deflator
                val_ozk = float(ozk[current_col]) * deflator
                val_nk = float(nk[current_col]) * deflator
                
                val_dv = float(donen_varlik[current_col]) * deflator if donen_varlik is not None else np.nan
                val_kvy = float(kisa_v_yukum[current_col]) * deflator if kisa_v_yukum is not None else np.nan
                val_satis = float(satis[current_col]) * deflator if satis is not None else np.nan
                
                val_nakit = float(nakit[current_col]) * deflator if nakit is not None else np.nan
                val_brut_kar = float(brut_kar[current_col]) * deflator if brut_kar is not None else np.nan
                val_faal_kar = float(faal_kar[current_col]) * deflator if faal_kar is not None else np.nan
                val_uzun_v_yukum = float(uzun_v_yukum[current_col]) * deflator if uzun_v_yukum is not None else np.nan
                val_stoklar = float(stoklar[current_col]) * deflator if stoklar is not None else np.nan
                val_maliyet = float(maliyet[current_col]) * deflator if maliyet is not None else np.nan
                
                # Hedef Değer - ENFLASYON DÜZELTMELİ
                next_tv = float(tv[next_col]) * next_deflator
                next_nk = float(nk[next_col]) * next_deflator
                
                if val_tv == 0 or val_ozk == 0 or next_tv == 0:
                    continue
                    
                roa = val_nk / val_tv
                roe = val_nk / val_ozk
                cari_oran = (val_dv / val_kvy) if (val_kvy and val_kvy != 0) else np.nan
                kar_marji = (val_nk / val_satis) if (val_satis and val_satis != 0) else np.nan
                
                nakit_orani = (val_nakit / val_kvy) if (val_kvy and val_kvy != 0) else np.nan
                brut_kar_marji = (val_brut_kar / val_satis) if (val_satis and val_satis != 0) else np.nan
                faal_kar_marji = (val_faal_kar / val_satis) if (val_satis and val_satis != 0) else np.nan
                kaldirac_uzun = (val_uzun_v_yukum / val_ozk) if (val_ozk and val_ozk != 0) else np.nan
                stok_devir = (val_maliyet / val_stoklar) if (val_stoklar and val_stoklar != 0) else np.nan
                
                target_roa = next_nk / next_tv
                
                data_list.append({
                    'Sirket': company,
                    'Donem': current_col,
                    'Hedef_Donem': next_col,
                    'ROA': roa,
                    'ROE': roe,
                    'Cari_Oran': cari_oran,
                    'Kar_Marji': kar_marji,
                    'Nakit_Orani': nakit_orani,
                    'Brut_Kar_Marji': brut_kar_marji,
                    'Faal_Kar_Marji': faal_kar_marji,
                    'Kaldirac_Uzun': kaldirac_uzun,
                    'Stok_Devir': abs(stok_devir) if not np.isnan(stok_devir) else np.nan,
                    'Enflasyon_Orani': cur_macro['enf'], 
                    'Dolar_Kuru': cur_macro['usd'],      
                    'Hedef_ROA': target_roa
                })
            except Exception as e:
                pass 
                
    except Exception as e:
        print(f"Hata {company}: {e}")

dataset = pd.DataFrame(data_list)
dataset.fillna(0, inplace=True)

# 2024 DAHİL VERİ SETİNİ KAYDET
dataset.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/tum_veri_seti_2024.xlsx', index=False)
print(f"\n2020-2024 Dev Veri Seti Boyutu: {dataset.shape}")

# BU KEZ 2024'ÜN İLK ÇEYREĞİNİ (2024/3) TAHMİN EDECEĞİZ!
# Bunun için model 2023/12 verilerine bakıp 2024/3'ü tahmin edecek.
train_df = dataset[~dataset['Donem'].str.contains('2023/12')]
test_df = dataset[dataset['Donem'].str.contains('2023/12')]

features = ['ROA', 'ROE', 'Cari_Oran', 'Kar_Marji', 'Nakit_Orani', 'Brut_Kar_Marji', 'Faal_Kar_Marji', 'Kaldirac_Uzun', 'Stok_Devir', 'Enflasyon_Orani', 'Dolar_Kuru']
target = 'Hedef_ROA'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"Eğitim Çeyrekleri: {len(X_train)} adet")
print(f"Test Çeyrekleri: {len(X_test)} adet")

print("\nXGBoost Modeli (2024 Tahmini İçin) Eğitiliyor...")
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- ENFLASYON DÜZELTMELİ (2024 İLK ÇEYREK) TAHMİNLERİ ---")
print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")

sonuc_tablosu = test_df[['Sirket', 'Hedef_Donem', 'Hedef_ROA']].copy()
sonuc_tablosu['Tahmin_Edilen_ROA'] = y_pred
sonuc_tablosu['Hata'] = sonuc_tablosu['Hedef_ROA'] - sonuc_tablosu['Tahmin_Edilen_ROA']

try:
    sonuc_tablosu.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/model_tahmin_2024.xlsx', index=False)
    print("\nSonuçlar 'model_tahmin_2024.xlsx' dosyasına kaydedildi.")
except PermissionError:
    print("\n[UYARI] Lütfen açık olan Excel dosyasını kapatın! Dosya üzerine yazılamadı.")
