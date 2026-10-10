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

BASE_CPI = macro_data.get('2020/3', {}).get('cpi', 1.0)

def get_row(df, keywords):
    for idx in df.index:
        if type(idx) == str:
            if any(kw.lower() in idx.lower() for kw in keywords):
                res = df.loc[idx]
                if isinstance(res, pd.DataFrame): return res.iloc[0]
                return res
    return None

data_list = []
print("2025 TAHMİN MODELİ: Listelediğiniz tüm yeni finansal oranlar hesaplanıyor...")

for company in companies:
    dfs = []
    # 2020'den 2026'ya kadar tüm yılları tara
    for year in range(2020, 2026):
        filepath = os.path.join(base_dir, company, f"{company}{year}.xlsx")
        if os.path.exists(filepath):
            df_year = pd.read_excel(filepath, index_col=0)
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
    brut_kar = get_row(df_combined, ["BRÜT KAR", "BRT KAR"])
    uzun_v_yukum = get_row(df_combined, ["Uzun Vadeli Yükümlülükler", "Uzun Vadeli"])
    stoklar = get_row(df_combined, ["Stoklar"])
    
    if tv is None or nk is None or ozk is None or satis is None:
        continue
        
    all_cols = list(df_combined.columns)
    
    for i in range(len(all_cols) - 1):
        current_col = all_cols[i]
        next_col = all_cols[i+1]
        
        cur_macro = macro_data.get(current_col)
        next_macro = macro_data.get(next_col)
        if not cur_macro or not next_macro: continue
        if 'cpi' not in cur_macro or 'cpi' not in next_macro: continue
            
        deflator = BASE_CPI / cur_macro['cpi'] 
        next_deflator = BASE_CPI / next_macro['cpi']
        
        try:
            val_tv = float(tv[current_col]) * deflator
            val_ozk = float(ozk[current_col]) * deflator
            val_nk = float(nk[current_col]) * deflator
            val_faal_kar = float(faal_kar[current_col]) * deflator if faal_kar is not None else np.nan
            val_satis = float(satis[current_col]) * deflator
            val_brut_kar = float(brut_kar[current_col]) * deflator if brut_kar is not None else np.nan
            
            val_dv = float(donen_varlik[current_col]) * deflator if donen_varlik is not None else np.nan
            val_kvy = float(kisa_v_yukum[current_col]) * deflator if kisa_v_yukum is not None else np.nan
            val_uzun_v = float(uzun_v_yukum[current_col]) * deflator if uzun_v_yukum is not None else np.nan
            val_stoklar = float(stoklar[current_col]) * deflator if stoklar is not None else np.nan
            
            next_tv = float(tv[next_col]) * next_deflator
            next_nk = float(nk[next_col]) * next_deflator
            
            if val_tv == 0 or val_ozk == 0 or next_tv == 0 or val_satis == 0: continue
            
            # SATIŞ BÜYÜMESİ (Bir önceki çeyreğe göre)
            satis_buyumesi = np.nan
            if i > 0:
                prev_col = all_cols[i-1]
                prev_macro = macro_data.get(prev_col)
                if prev_macro:
                    prev_deflator = BASE_CPI / prev_macro['cpi']
                    prev_satis = float(satis[prev_col]) * prev_deflator
                    if prev_satis != 0:
                        satis_buyumesi = (val_satis - prev_satis) / prev_satis

            # YENİ FİNANSAL ORANLAR KÜMESİ (Kârlılık, Likidite, Kaldıraç, Verimlilik, Büyüme)
            roa = val_nk / val_tv
            roe = val_nk / val_ozk
            net_kar_marji = val_nk / val_satis
            brut_kar_marji = (val_brut_kar / val_satis) if not np.isnan(val_brut_kar) else np.nan
            faaliyet_kar_marji = (val_faal_kar / val_satis) if not np.isnan(val_faal_kar) else np.nan
            
            cari_oran = (val_dv / val_kvy) if (val_kvy and val_kvy != 0) else np.nan
            asit_test = ((val_dv - (val_stoklar if not np.isnan(val_stoklar) else 0)) / val_kvy) if (val_kvy and val_kvy != 0) else np.nan
            
            toplam_borc = (val_kvy if not np.isnan(val_kvy) else 0) + (val_uzun_v if not np.isnan(val_uzun_v) else 0)
            borc_ozkaynak = toplam_borc / val_ozk
            borc_aktif = toplam_borc / val_tv
            
            aktif_devir_hizi = val_satis / val_tv
            
            target_roa = next_nk / next_tv
            
            ay_str = current_col.split('/')[1]
            on_ikinci_ay_mi = 1 if float(ay_str) == 12 else 0
            
            data_list.append({
                'Sirket': company,
                'Donem': current_col,
                'Hedef_Donem': next_col,
                'ROA': roa,
                'ROE': roe,
                'Net_Kar_Marji': net_kar_marji,
                'Brut_Kar_Marji': brut_kar_marji,
                'Faaliyet_Kar_Marji': faaliyet_kar_marji,
                'Cari_Oran': cari_oran,
                'Asit_Test': asit_test,
                'Borc_Ozkaynak': borc_ozkaynak,
                'Borc_Aktif': borc_aktif,
                'Aktif_Devir_Hizi': aktif_devir_hizi,
                'Satis_Buyumesi': satis_buyumesi,
                'Enflasyon_Orani': cur_macro['enf'], 
                'Dolar_Kuru': cur_macro['usd'],
                'On_Ikinci_Ay_Mi': on_ikinci_ay_mi,
                'Hedef_ROA': target_roa
            })
        except Exception as e:
            pass 

dataset = pd.DataFrame(data_list)
dataset.fillna(0, inplace=True)

dataset.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/tum_veri_seti_2025_YeniOranlar.xlsx', index=False)
print(f"Yeni Oranlarla Dev Veri Seti Boyutu: {dataset.shape}")

# Modelin 2024/12'ye bakıp 2025/3'ü tahmin etmesini istiyoruz. 
test_df = dataset[dataset['Donem'].str.contains('2024/12')]
train_df = dataset[~dataset['Donem'].str.contains('2024/12')]

# Eğer kullanıcının klasöründe henüz 2025 bilançoları yoksa (yani Hedef_Donem = 2025/3 oluşmadıysa)
# Kodun çökmemesi için otomatik olarak bir önceki dönemi test ederiz.
if len(test_df) == 0:
    print("\n[BİLGİ] Klasörünüzde henüz '2025' yılına ait Excel dosyaları bulunamadığı için 2025/3 hedefi oluşturulamadı.")
    print("Bunun yerine model 2024/9 verilerine bakarak 2024 Yıl Sonu (2024/12) tahmin sınavına sokuluyor...")
    test_df = dataset[dataset['Donem'].str.contains('2024/9')]
    train_df = dataset[~dataset['Donem'].str.contains('2024/9')]

features = [
    'ROA', 'ROE', 'Net_Kar_Marji', 'Brut_Kar_Marji', 'Faaliyet_Kar_Marji', 
    'Cari_Oran', 'Asit_Test', 'Borc_Ozkaynak', 'Borc_Aktif', 'Aktif_Devir_Hizi', 
    'Satis_Buyumesi', 'Enflasyon_Orani', 'Dolar_Kuru', 'On_Ikinci_Ay_Mi'
]
target = 'Hedef_ROA'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"Eğitim Çeyrekleri: {len(X_train)} adet")
print(f"Test Çeyrekleri: {len(X_test)} adet")

if len(X_test) > 0:
    print("\nXGBoost Modeli (Yeni Süper Finansal Oranlarla) Eğitiliyor...")
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n--- YENİ ORANLARLA MODEL SONUÇLARI ---")
    print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")

    sonuc_tablosu = test_df[['Sirket', 'Hedef_Donem', 'Hedef_ROA']].copy()
    sonuc_tablosu['Tahmin_Edilen_ROA'] = y_pred
    sonuc_tablosu['Hata'] = sonuc_tablosu['Hedef_ROA'] - sonuc_tablosu['Tahmin_Edilen_ROA']

    try:
        sonuc_tablosu.to_excel('c:/Users/faruk/PycharmProjects/TübitakDeneme/model_tahmin_YeniOranlar.xlsx', index=False)
        print("\nSonuçlar 'model_tahmin_YeniOranlar.xlsx' dosyasına kaydedildi.")
    except PermissionError:
        print("\n[UYARI] Lütfen açık olan Excel dosyasını kapatın! Dosya üzerine yazılamadı.")
else:
    print("Test edilecek çeyrek bulunamadı. Lütfen klasörleri kontrol edin.")
