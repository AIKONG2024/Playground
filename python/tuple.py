# 파이썬 튜플

# 리스트와 비교 중요
# 튜플 자료형 (순서0 , 중복0, 수정x, 삭제 x) #불변. 한번 선언하면 끝까지 사용.

# 선언
a = ()
b = 1
b2 = (1,)
c = (11, 12, 13, 14)
d = (100, 1000, "Ace", "Base", "Captine")
e = (100, 1000, ("Ace", "Base", "Captine"))

print(type(a), type(b))  # <class 'tuple'> <class 'int'> 타입 출력
print(type(b2))  # <class 'tuple'> 원소가 하나 있을땐 점(,)을 찍어야 튜플로 인식.

# 인덱싱
print(">>>>>>>")
print(" d - ", d[1])  # d - 1000 튜플의 1번째 인덱스의 값을 가져옴
print(" d - ", d[0] - d[1] - d[1])  # d - -1900 /가져온 데이터의 연산
print(" d -", d[-1])  # d - Captine / -1은 튜플 맨뒤의 값
print(" d -", e[-1])  # d - ('Ace', 'Base', 'Captine')
print(" d -", e[-1][1])  # d - Base
# 튜플을 리스트로 변환하면 불변이라는 특징이 사라지고 수정과 삭제가 가능해짐.
print(list(e[-1][1]))  # ['B', 'a', 's', 'e']

# 수정
# d[0] = 3500 #오류 발생 튜플은 수정 불가

# 슬라이싱
print(">>>>>")
print("d - ", d[0:3])  # d -  (100, 1000, 'Ace') 0~3의 데이터를 가져옴.
print("d - ", d[2:])  # d -  ('Ace', 'Base', 'Captine')
print("d -", e[2][1:3])  # d - ('Base', 'Captine')

# 튜플 연산
print(">>>>>")
print("c + d", c + d)  # c + d (11, 12, 13, 14, 100, 1000, 'Ace', 'Base', 'Captine')
print("c * 3", c * 3)  # c * 3 (11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14

# 튜플 함수
a = (5, 2, 3, 1, 4)
print("a - ", a)  # a -  (5, 2, 3, 1, 4)
print("a - ", a.index(3))  # a -  2 index로 값을 빼올수 있음.
print("a - ", a.count(3))  # a -  1 해당 원소의 갯수 반환도 가능.

# =====팩킹 & 언팩킹=====
# 팩킹 : 하나로 묶는것
t = ("foo", "bar", "baz", "qux")
print(t)  # ('foo', 'bar', 'baz', 'qux')
print(t[0])  # foo  하나로 묶어서 인덱스로 값을 빼올 수 있음.

# 언팩킹1 - 튜플로 묶여진 값을 풀어서 가져옴.
(x1, x2, x3, x4) = t  # x1, x2, x3, x4 괄호가 없어도 똑같음. == x1, x2, x3, x4 = t

print(
    type(x1), type(x2), type(x3), type(x4)
)  # <class 'str'> <class 'str'> <class 'str'> <class 'str'>
print(x1, x2, x3, x4)  # foo bar baz qux

# 팩킹 & 언팩킹
t2 = 1, 2, 3  # 괄호가 없어도 튜플 형식임. 팩킹
t3 = (4,) 
x1, x2, x3 = t2 #언팩킹
x4, x5, x6 = 4, 5, 6 
print(t2, t3)  # (1, 2, 3) (4,)
print(x1, x2, x3, x4, x5, x6)  # 1 2 3 4 5 6
