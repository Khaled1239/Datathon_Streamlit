import geopandas as gpd
import fiona

# ==== 1. Path ke file GDB ====
gdb_path = "Kotaw.gdb"  # ganti sesuai nama folder GDB kamu

# ==== 2. Cek semua layer ====
layers = fiona.listlayers(gdb_path)
print("Layer yang tersedia:", layers)

# ==== 3. Baca layer kabupaten/kota ====
# (dari hasil listlayers biasanya 'ADMINISTRASI_AR_KABKOTA')
gdf = gpd.read_file(gdb_path, layer="ADMINISTRASI_AR_KABKOTA")

print("\nKolom yang tersedia di layer:")
print(gdf.columns)

# ==== 4. Cari kolom yang mengandung 'JAWA TIMUR' ====
target_col = None
for col in gdf.columns:
    if gdf[col].astype(str).str.contains("JAWA TIMUR", case=False, na=False).any():
        target_col = col
        break

if target_col is None:
    raise ValueError("❌ Tidak ditemukan kolom dengan teks 'JAWA TIMUR'. Cek isi atribut tabel!")

print(f"\nKolom yang dipakai untuk filter: {target_col}")

# ==== 5. Filter hanya Jawa Timur ====
gdf_jatim = gdf[gdf[target_col].str.contains("JAWA TIMUR", case=False, na=False)]

print(f"Jumlah fitur Jawa Timur: {len(gdf_jatim)}")

# ==== 6. Simpan ke GeoJSON ====
output_file = "Main.geojson"
gdf_jatim.to_file(output_file, driver="GeoJSON")
print(f"✅ Data Jawa Timur berhasil disimpan ke {output_file}")
