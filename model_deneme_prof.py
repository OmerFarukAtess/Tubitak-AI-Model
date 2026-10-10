import pandas as pd
import numpy as np
import os
import traceback

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_dir = r'c:\Users\faruk\PycharmProjects\TübitakDeneme\T-bitakVerileri\tübitak veriler'
companies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# MAKRO VERİLER
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
BASE_CPI = 450.0

def get_row(df, keywords):
    for idx in df.index:
        if type(idx) == str:
            if any(kw.lower() in idx.lower() for kw in keywords):
                res = df.loc[idx]
                if isinstance(res, pd.DataFrame): return res.iloc[0]
                return res
    return None

data_list = []
print("PROFESYONEL MODEL: Faaliyet Kârı ve 12. Ay İşaretçisi (Flag) ile veriler birleştiriliyor...")

for company in companies:
    # 2020-2024 Dosyaları Döngüsü
    dfs = []
    for year in range(2020, 2025):
        filepath = os.path.join(base_dir, company, f"{company}{year}.xlsx")
        if os.path.exists(filepath):
            df_year = pd.read_excel(filepath, index_col=0)
            
            # KCHOL Hatasını (3.1) Çözmek İçin int(float(...)) kullanıyoruz!
            cols = sorted([c for c in df_year.columns if type(c) == str and f'{year}/' in c], 
                          key=lambda x: int(float(x.split('/')[1])))
            dfs.append(df_year[cols])
            
    if not dfs: continue
    
    df_combined = pd.concat(dfs, axis=1)
    
    tv = get_row(df_combined, ["TOPLAM VARLIKLAR"])
    ozk = get_row(df_combined, ["zkaynaklar", "Özkaynaklar"])
    nk = get_row(df_combined, ["NEM KARI (ZARARI)", "DÖNEM KARI"])
    faal_kar = get_row(df_combined, ["net faaliyet", "esas faaliyet"])
    donen_varlik = get_row(df_combined, ["Dönen Varlıklar", "Dnen Varlklar"])
    kisa_v_yukum = get_row(df_combined, ["Kısa Vadeli Yükümlülükler", "Ksa Vadeli"])
    satis = get_row(df_combined, ["Satış Gelirleri", "Satş Gelirleri"])
    nakit = get_row(df_combined, ["Nakit ve Nakit Benzerleri", "Nakit"])
    brut_kar = get_row(df_combined, ["BRÜT KAR", "BRT KAR"])
    uzun_v_yukum = get_row(df_combined, ["Uzun Vadeli Yükümlülükler", "Uzun Vadeli"])
    stoklar = get_row(df_combined, ["Stoklar"])
    maliyet = get_row(df_combined, ["Satışların Maliyeti", "Satşlarn Maliyeti"])
    
    if tv is None or faal_kar is None or ozk is None:
        continue
        
    all_cols = list(df_combined.columns)
    
    for i in range(len(all_cols) - 1):
        current_col = all_cols[i]
        next_col = all_cols[i+1]
        
        cur_macro = macro_data.get(current_col)
        next_macro = macro_data.get(next_col)
        if not cur_macro or not next_macro: continue
            
        deflator = BASE_CPI / cur_macro['cpi'] 
        next_deflator = BASE_CPI / next_macro['cpi']
        
        try:
            val_tv = float(tv[current_col]) * deflator
            val_ozk = float(ozk[current_col]) * deflator
            val_nk = float(nk[current_col]) * deflator if nk is not None else np.nan
            val_faal_kar = float(faal_kar[current_col]) * deflator
            
            val_dv = float(donen_varlik[current_col]) * deflator if donen_varlik is not None else np.nan
            val_kvy = float(kisa_v_yukum[current_col]) * deflator if kisa_v_yukum is not None else np.nan
            val_satis = float(satis[current_col]) * deflator if satis is not None else np.nan
            val_nakit = float(nakit[current_col]) * deflator if nakit is not None else np.nan
            val_brut_kar = float(brut_kar[current_col]) * deflator if brut_kar is not None else np.nan
            val_uzun_v_yukum = float(uzun_v_yukum[current_col]) * deflator if uzun_v_yukum is not None else np.nan
            val_stoklar = float(stoklar[current_col]) * deflator if stoklar is not None else np.nan
            val_maliyet = float(maliyet[current_col]) * deflator if maliyet is not None else np.nan
            
            next_tv = float(tv[next_col]) * next_deflator
            next_faal_kar = float(faal_kar[next_col]) * next_deflator
            
            if val_tv == 0 or val_ozk == 0 or next_tv == 0: continue
            
            # YENİ YÖNTEM: Gerçek ROA (Net Kâr yerine Faaliyet Kârı)
            gercek_roa = val_faal_kar / val_tv
            roe = val_faal_kar / val_ozk
            cari_oran = (val_dv / val_kvy) if (val_kvy and val_kvy != 0) else np.nan
            
            nakit_orani = (val_nakit / val_kvy) if (val_kvy and val_kvy != 0) else np.nan
            brut_kar_marji = (val_brut_kar / val_satis) if (val_satis and val_satis != 0) else np.nan
            faal_kar_marji = (val_faal_kar / val_satis) if (val_satis and val_satis != 0) else np.nan
            kaldirac_uzun = (val_uzun_v_yukum / val_ozk) if (val_ozk and val_ozk != 0) else np.nan
            stok_devir = (val_maliyet / val_stoklar) if (val_stoklar and val_stoklar != 0) else np.nan
            
            target_gercek_roa = next_faal_kar / next_tv
            
            # 12. AY İŞARETÇİSİ (FLAG)
            ay_str = current_col.split('/')[1]
            on_ikinci_ay_mi = 1 if float(ay_str) == 12 else 0
            
            data_list.append({
                'Sirket': company,
                'Donem': current_col,
                'Hedef_Donem': next_col,
                'Gercek_ROA': gercek_roa,
                'Gercek_ROE': roe,
                'Cari_Oran': cari_oran,
                'Nakit_Orani': nakit_orani,
                'Brut_Kar_Marji': brut_kar_marji,
                'Faal_Kar_Marji': faal_kar_marji,
                'Kaldirac_Uzun': kaldirac_uzun,
                'Stok_Devir': abs(stok_devir) if not np.isnan(stok_devir) else np.nan,
                'Enflasyon_Orani': cur_macro['enf'], 
                'Dolar_Kuru': cur_macro['usd'],
                'On_Ikinci_Ay_Mi': on_ikinci_ay_mi, # YENİ ZEKİ SÜTUN!
                'Hedef_Gercek_ROA': target_gercek_roa
            })
        except Exception as e:
            pass 

dataset = pd.DataFrame(data_list)
dataset.fillna(0, inplace=True)

dataset.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/tum_veri_seti_Prof.xlsx', index=False)
print(f"\nProfesyonel Veri Seti Boyutu: {dataset.shape} (KCHOL Başarıyla Eklendi!)")

train_df = dataset[~dataset['Donem'].str.contains('2023/12')]
test_df = dataset[dataset['Donem'].str.contains('2023/12')]

# ÖZELLİKLER ARASINA "On_Ikinci_Ay_Mi" EKLENDİ
features = ['Gercek_ROA', 'Gercek_ROE', 'Cari_Oran', 'Nakit_Orani', 'Brut_Kar_Marji', 'Faal_Kar_Marji', 'Kaldirac_Uzun', 'Stok_Devir', 'Enflasyon_Orani', 'Dolar_Kuru', 'On_Ikinci_Ay_Mi']
target = 'Hedef_Gercek_ROA'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"Eğitim Çeyrekleri: {len(X_train)} adet")
print(f"Test Çeyrekleri (2024/03 Sınavı): {len(X_test)} adet")

print("\nXGBoost (Profesyonel) Modeli Eğitiliyor...")
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- PROFESYONEL MODEL SONUÇLARI (2024 İLK ÇEYREK TAHMİNLERİ) ---")
print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")

sonuc_tablosu = test_df[['Sirket', 'Hedef_Donem', 'Hedef_Gercek_ROA']].copy()
sonuc_tablosu['Tahmin_Edilen_Gercek_ROA'] = y_pred
sonuc_tablosu['Hata'] = sonuc_tablosu['Hedef_Gercek_ROA'] - sonuc_tablosu['Tahmin_Edilen_Gercek_ROA']

try:
    sonuc_tablosu.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/model_tahmin_2024_Prof.xlsx', index=False)
    print("\nSonuçlar 'model_tahmin_2024_Prof.xlsx' dosyasına kaydedildi.")
except PermissionError:
    print("\n[UYARI] Lütfen açık olan Excel dosyasını kapatın! Dosya üzerine yazılamadı.")
