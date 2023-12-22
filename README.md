# Energy Forecasting via Deep Machine Learning

> Univariate Time-Series Energy Forecasting with LSTM

#### Created and Modified by Dr.PEACE | 22/12/2023

##### ในไฟล์ประกอบด้วย
* ไฟล์ *1_Version-Test-Check.py* ใช้สำหรับตรวจสอบ Version ของ Library ต่าง ๆ ว่ามีครบแล้วหรือไม่
* ไฟล์ *2_Univariate_Data_Plot.py* ใช้สำหรับพล็อตกราฟค่าที่จะนำไป Train และ Test ของไฟล์นั้น ๆ
* ไฟล์ *3_LSTM-Univariate.py* ใช้สำหรับรันโมเดล LSTM พยากรณ์หาค่าพลังงานไฟฟ้า

##### โฟลเดอร์ที่เก็บไฟล์
* โฟลเดอร์ *dataset* ไว้สำหรับเก็บชุดข้อมูลที่จะรัน
* โฟลเดอร์ *figure* ไว้เก็บรูปภาพต่าง ๆ
* โฟลเดอร์ *result* ไว้เก็บผลลัพธ์ที่ได้จากการรัน

##### สำหรับไฟล์ *3_LSTM-Univariate.py*
> แก้ไขบรรทัดที่ 83-86

```python
filename = 'dataset/Bangkok_solarpv_Trial.csv' # ชื่อไฟล์ที่จะนำมาพยากรณ์
horizon = 372 # ค่าสเตปการพยากรณ์ ดูเพิ่มเติมได้จาก dataset details.pdf
neurons = 10 # ค่าโหนดสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 10)
epoch = 100 # ค่าการวนซ้ำสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 100)
```

##### สำหรับไฟล์ *Recapture.py*
> ไว้สำหรับทบทวนสิ่งที่ได้เรียนไปเกี่ยวกับ Python พื้นฐานการใช้งานต่าง ๆ