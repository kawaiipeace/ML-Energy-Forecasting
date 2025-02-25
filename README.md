# Energy Forecasting via Deep Machine Learning

> Univariate Time-Series Energy Forecasting with LSTM by Dr.PEACE | 25/02/2025

## สิ่งที่จำเป็นก่อนเริ่ม (Prerequisite) สำหรับระบบปฏิบัติการ Windows
1. เปิด Microsoft PowerShell (ใช้สิทธิ์ Administrator) ทำการติดตั้ง chocolatey
```bash
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
2. ทำการติดตั้ง pyenv
```bash
choco install pyenv-win
```
3. ทำการกำหนด Execution Policy เพื่อให้ใช้งาน pyenv ได้
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
```
4. ทำการติดตั้ง python 3.10.5
```bash
pyenv install 3.10.5
```
5. ตั้งค่า Global Environment ให้ python เป็น 3.10.5
```bash
pyenv global 3.10.5
```
6. ติดตั้ง Dependency สำหรับใช้ใน Project นี้ ผ่าน Terminal ใน VSCode
```bash
pip install -r requirements.txt
```

## ในไฟล์ประกอบด้วย
* ไฟล์ *1_Version-Test-Check.py* ใช้สำหรับตรวจสอบ Version ของ Library ต่าง ๆ ว่ามีครบแล้วหรือไม่
* ไฟล์ *2_Univariate_Data_Plot.py* ใช้สำหรับพล็อตกราฟค่าที่จะนำไป Train และ Test ของไฟล์นั้น ๆ
* ไฟล์ *3_LSTM-Univariate.py* ใช้สำหรับรันโมเดล LSTM พยากรณ์หาค่าพลังงานไฟฟ้า

##### โฟลเดอร์ที่เก็บไฟล์
* โฟลเดอร์ *dataset* ไว้สำหรับเก็บชุดข้อมูลที่จะรัน
* โฟลเดอร์ *figure* ไว้เก็บรูปภาพต่าง ๆ
* โฟลเดอร์ *result* ไว้เก็บผลลัพธ์ที่ได้จากการรัน

##### สำหรับไฟล์ *3_LSTM-Univariate.py*
> แก้ไขบรรทัดที่ 23-28

```python
filename = 'dataset/Bangkok_solarpv_Trial.csv' # ชื่อไฟล์ที่จะนำมาพยากรณ์
horizon = 372 # ค่าสเตปการพยากรณ์ ดูเพิ่มเติมได้จาก dataset details.pdf
neurons = 10 # ค่าโหนดสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 10)
epoch = 100 # ค่าการวนซ้ำสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 100)
```