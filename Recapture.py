## Recapture ทบทวนสิ่งที่เรียนเมื่อ 23/12/2566

# ฟังก์ชั่นการบวกเลข 2 ตัว
def add_cal(x,y):
    return x+y

# ฟังก์ชั่นการลบเลข 2 ตัว
def sub_cal(x,y):
    return x-y

# ฟังก์ชั่นการคูณเลข 2 ตัว
def mul_cal(x,y):
    return x*y

# ฟังก์ชั่นการหารเลข 2 ตัว
def div_cal(x,y):
    return x/y

# ฟังก์ชั่นการตัดเกรด
def grade_cal(score):
    if score >= 80:
        print("เกรดของคุณคือ: Excellent")
    elif score >= 60 and score <=79:
        print("เกรดของคุณคือ: Good")
    elif score >= 50 and score <=59:
        print("เกรดของคุณคือ: Pass")
    else:
        print("เกรดของคุณคือ: Fail")

# ฟังก์ชั่นการหาสูตรคูณแบบ While Loops
def mul_table_whileloops(number,mul):
    print('สูตรคูณแม่',mul)
    count = 1
    while count <= number:
        print(mul,'x',count,'=',mul*count)
        count +=1
    else:
        print('- จบการคูณแบบ While Loops -\n')

# ฟังก์ชั่นการหาสูตรคูณแบบ For Loops    
def mul_table_forloops(number,mul):
    print('สูตรคูณแม่',mul)
    for count in range(1,number+1):
        print(mul,'x',count,'=',mul*count)
    else:
        print('- จบการคูณแบบ For Loops -\n')

# Main Begin Here
print('ยินดีต้อนรับสู่การทบทวนบทเรียนต่าง ๆ ที่เรียนในวันนี้ (23/12/2566)\n')
print('รายการโปรแกรม\n')
print('เลือก 1 โปรแกรมบวกเลข 2 หลัก\n')
print('เลือก 2 โปรแกรมลบเลข 2 หลัก\n')
print('เลือก 3 โปรแกรมคูณเลข 2 หลัก\n')
print('เลือก 4 โปรแกรมหารเลข 2 หลัก\n')
print('เลือก 5 โปรแกรมบวกลบคูณหารและยกกำลัง 2 หลักแบบ Lambda Function\n')
print('เลือก 6 โปรแกรมตัดเกรด\n')
print('เลือก 7 โปรแกรมตารางสูตรคูณแบบ While Loops\n')
print('เลือก 8 โปรแกรมตารางสูตรคูณแบบ For Loops\n')
print('-------------------------\n')

select_var = input('กรุณาเลือกเมนูที่ต้องการ (กรอกเป็นตัวเลข): ')
print('ท่านเลือกเมนูที่',select_var)
print('-------------------------\n')
if select_var == "1":
    x = int(input('กรอกตัวเลขแรก:'))
    y = int(input('กรอกตัวเลขสอง:'))
    total = add_cal(x,y)
    print('ผลลัพธ์ของการคำนวณคือ ',total)
elif select_var == "2":
    x = int(input('กรอกตัวเลขแรก:'))
    y = int(input('กรอกตัวเลขสอง:'))
    total = sub_cal(x,y)
    print('ผลลัพธ์ของการคำนวณคือ ',total)
elif select_var == "3":
    x = int(input('กรอกตัวเลขแรก:'))
    y = int(input('กรอกตัวเลขสอง:'))
    total = mul_cal(x,y)
    print('ผลลัพธ์ของการคำนวณคือ ',total)
elif select_var == "4":
    x = int(input('กรอกตัวเลขแรก:'))
    y = int(input('กรอกตัวเลขสอง:'))
    total = div_cal(x,y)
    print('ผลลัพธ์ของการคำนวณคือ ',total)
elif select_var == "5":
    x = int(input('กรอกตัวเลขแรก:'))
    y = int(input('กรอกตัวเลขสอง:'))
    total_add = lambda x, y: x+y
    total_sub = lambda x, y: x-y
    total_mul = lambda x, y: x*y
    total_div = lambda x, y: x/y
    total_sqr = lambda x, y: x**y
    print('ผลลัพธ์ของการบวกคือ ',total_add(x,y))
    print('ผลลัพธ์ของการลบคือ ',total_sub(x,y))
    print('ผลลัพธ์ของการคูณคือ ',total_mul(x,y))
    print('ผลลัพธ์ของการหารคือ ',total_div(x,y))
    print('ผลลัพธ์ของการยกกำลังคือ ',total_sqr(x,y))
elif select_var == "6":
    name = str(input('กรุณากรอกชื่อ:'))
    score = int(input('กรุณากรอกคะแนน:'))
    print('คุณชื่อ',name)
    print('ได้คะแนน',score)
    grade_cal(score)
elif select_var == "7":
    number = int(input('กรอกจำนวนครั้งที่จะคูณ:'))
    mul = int(input('กรอกเลขแม่สูตรคูณ:'))
    mul_table_whileloops(number,mul)
elif select_var == "8":
    number = int(input('กรอกจำนวนครั้งที่จะคูณ:'))
    mul = int(input('กรอกเลขแม่สูตรคูณ:'))
    mul_table_forloops(number,mul)
else:
    print('โปรดกรอกตัวเลขระหว่าง 1-8 เท่านั้น')