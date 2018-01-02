print("Populasi Lebah")
import random
jmlLebah = random.randint(40000,60000)
print(jmlLebah)
Jantan = random.randint(1000,1500)
Betina = jmlLebah - Jantan - 1
print("jumlah jantan :",Jantan)
print("jumlah betina :",Betina)
telur = []
larva = []
pupa = []
betina = []
jantan = []
for i in range(6000):
  telur.append(random.randint(0,2))
for i in range(10000):
  larva.append(random.randint(0,6))
for i in range(20000):
  pupa.append(random.randint(0,6))
for i in range(Betina):
  betina.append(random.randint(0,4))
for i in range(Jantan):
  jantan.append(random.randint(0,7))
WP = input("Lama waktu pengujian simulasi dalam hari : ")
adaHama = raw_input("Apakah ada tabuhan disekitar lingkungan simulasi?(y/n)")
if adaHama == 'y':
  print("Keberadaan hama tabuhan dikonfirmasi")
else:
  print("Tidak ada hama tabuhan")
