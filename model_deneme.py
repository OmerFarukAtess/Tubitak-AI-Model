import pandas as pd
import numpy as np
import os
import traceback

# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_dir = r'c:\Users\faruk\PycharmProjects\TübitakDeneme\T-bitakVerileri\tübitak veriler'
companies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

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
print("2020, 2021 ve 2022 yılları birleştirilerek daha büyük bir veri seti oluşturuluyor...")

for company in companies:
    # 2020, 2021 ve 2022 dosyalarının yolları
    file_2020 = os.path.join(base_dir, company, f"{company}2020.xlsx")
    file_2021 = os.path.join(base_dir, company, f"{company}2021.xlsx")
    file_2022 = os.path.join(base_dir, company, f"{company}2022.xlsx")
    
    if not (os.path.exists(file_2020) and os.path.exists(file_2021) and os.path.exists(file_2022)):
        continue
        
    try:
        df_2020 = pd.read_excel(file_2020, index_col=0)
        df_2021 = pd.read_excel(file_2021, index_col=0)
        df_2022 = pd.read_excel(file_2022, index_col=0)
        
        # Sütunları ayıkla ve sayısal sırala (3, 6, 9, 12)
        cols_2020 = sorted([c for c in df_2020.columns if type(c) == str and '2020/' in c], key=lambda x: int(x.split('/')[1]))
        cols_2021 = sorted([c for c in df_2021.columns if type(c) == str and '2021/' in c], key=lambda x: int(x.split('/')[1]))
        cols_2022 = sorted([c for c in df_2022.columns if type(c) == str and '2022/' in c], key=lambda x: int(x.split('/')[1]))
        
        # Üç yılı (toplam 12 çeyrek) birleştir
        df_combined = pd.concat([df_2020[cols_2020], df_2021[cols_2021], df_2022[cols_2022]], axis=1)
        
        # Satırları bul
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
            
        all_cols = cols_2020 + cols_2021 + cols_2022
        
        for i in range(len(all_cols) - 1):
            current_col = all_cols[i]
            next_col = all_cols[i+1]
            
            try:
                # Girdi Değerleri
                val_tv = float(tv[current_col])
                val_ozk = float(ozk[current_col])
                val_nk = float(nk[current_col])
                val_dv = float(donen_varlik[current_col]) if donen_varlik is not None else np.nan
                val_kvy = float(kisa_v_yukum[current_col]) if kisa_v_yukum is not None else np.nan
                val_satis = float(satis[current_col]) if satis is not None else np.nan
                
                val_nakit = float(nakit[current_col]) if nakit is not None else np.nan
                val_brut_kar = float(brut_kar[current_col]) if brut_kar is not None else np.nan
                val_faal_kar = float(faal_kar[current_col]) if faal_kar is not None else np.nan
                val_uzun_v_yukum = float(uzun_v_yukum[current_col]) if uzun_v_yukum is not None else np.nan
                val_stoklar = float(stoklar[current_col]) if stoklar is not None else np.nan
                val_maliyet = float(maliyet[current_col]) if maliyet is not None else np.nan
                
                # Hedef Değer
                next_tv = float(tv[next_col])
                next_nk = float(nk[next_col])
                
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
                    'Hedef_ROA': target_roa
                })
            except Exception as e:
                pass 
                
    except Exception as e:
        print(f"Hata {company}: {e}")

dataset = pd.DataFrame(data_list)
dataset.fillna(0, inplace=True)

# GÖRSELLEŞTİRME İÇİN VERİ SETİNİ KAYDEDELİM
dataset.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/tum_veri_seti.xlsx', index=False)
print("\n[BİLGİ] Veri setinin tam hali 'tum_veri_seti.xlsx' olarak kaydedildi.")

print(f"\n3 Yıllık (2020-2022) Büyük Veri Seti Boyutu: {dataset.shape}")

# EĞİTİM VE TEST AYRIMI
# Bu kez hedefimiz 2022'nin ilk çeyreğini (2022/3) tahmin etmek.
# O yüzden 2021/12 satırlarını test için ayırıyoruz. Geri kalan her şey (2020'nin tamamı ve 2021'in ilk 3 çeyreği) Eğitim olacak.
train_df = dataset[~dataset['Donem'].str.contains('2021/12')]
test_df = dataset[dataset['Donem'].str.contains('2021/12')]

features = ['ROA', 'ROE', 'Cari_Oran', 'Kar_Marji', 'Nakit_Orani', 'Brut_Kar_Marji', 'Faal_Kar_Marji', 'Kaldirac_Uzun', 'Stok_Devir']
target = 'Hedef_ROA'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"\nModelin Öğrendiği Çeyrek Sayısı (Eğitim): {len(X_train)}")
print(f"Modelin Sınava Girdiği Çeyrek Sayısı (Test - 2022/3): {len(X_test)}")

print("\nXGBoost Modeli Eğitiliyor...")
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- 2022 İLK ÇEYREK TAHMİN SONUÇLARI ---")
print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")

sonuc_tablosu = test_df[['Sirket', 'Hedef_Donem', 'Hedef_ROA']].copy()
sonuc_tablosu['Tahmin_Edilen_ROA'] = y_pred
sonuc_tablosu['Hata'] = sonuc_tablosu['Hedef_ROA'] - sonuc_tablosu['Tahmin_Edilen_ROA']

print("\n--- ŞİRKET BAZINDA 2022 TAHMİNLERİ ---")
print(sonuc_tablosu.to_string(index=False))

try:
    sonuc_tablosu.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/model_tahmin_2022.xlsx', index=False)
    print("\nSonuçlar 'model_tahmin_2022.xlsx' dosyasına kaydedildi.")
except PermissionError:
    print("\n[UYARI] Lütfen açık olan Excel dosyasını kapatın! Dosya üzerine yazılamadı.")
