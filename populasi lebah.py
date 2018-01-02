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
Luas = input("Luas lahan kebun stroberi dalam are : ")
adaHama = raw_input("Apakah ada tabuhan disekitar lingkungan simulasi?(y/n)")
LStroberi = int(Betina*4.96/100)
print("Jumlah lebah yang menuju Stroberi : ",LStroberi)
jmlTanaman = Luas*1000000/625
print("Jumlah tanaman stroberi : ",jmlTanaman)
BungaTanaman = []
jmlBunga = 0
for i in range(jmlTanaman):
  temp = random.randint(0,5)
  BungaTanaman.append(temp)
  jmlBunga = jmlBunga + temp
print("Jumlah bunga pada lahan : ",jmlBunga)
WPanen = random.randint(3,7)
print("Waktu panen saat ini : ",WPanen)
