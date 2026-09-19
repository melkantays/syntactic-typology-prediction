import lang2vec.lang2vec as l2v

# Elamca'nin ISO kodu "elx". Karsilastirma icin iki bilinen dil de ekliyoruz.
diller = ["elx", "eus", "tur"]   # elx=Elamca, eus=Baskca (izolat), tur=Turkce
setler = ["syntax_knn", "syntax_wals", "syntax_sswl", "syntax_average"]

for dil in diller:
    print(f"\n===== {dil.upper()} =====")
    for s in setler:
        try:
            f = l2v.get_features([dil], s)[dil]
            # "--" isareti = o ozellik icin veri YOK demek
            dolu = sum(1 for v in f if v != "--")
            print(f"{s:16s} uzunluk={len(f):4d}  dolu={dolu:4d}  bos={len(f)-dolu:4d}")
        except Exception as e:
            print(f"{s:16s} HATA: {e}")