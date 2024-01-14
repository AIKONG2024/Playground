# dictionary


# 범용적으로 실무에서 가장 많이 사용.
# json 형태로 많이 쓰임. (csv 파일과 비슷한 데이터 파일)
# 딕셔너리 자료형 (순서x, 키 중복x, 수정o , 삭제 o)

# 선언
# a = {'name' : 'Kim', 'name': 'Lee'} #오류, 키 중복 허용안됨,
a = {'name': 'Kim', 'phone' : '01033337777', 'birth' : '870514'} #key는 인트형도 가능. 보통 문자형으로 사용
b = {0 : 'Hello Python'}
c = {'arr': [1,2,3,4]} #키만 존재하면 모든 데이터형 사용이 가능.
d = {
    'Name' : 'Niceman',
    'City' : 'Seoul',
    'Age' : 33,
    'Grade' : 'A',
    'Status' : True
}
 #dict 안에 리스트의 튜플로 넣을 수도 있음.
e = dict([
    ('Name', 'Niceman'),
    ('City', 'Seoul'),
    ('Age', 33),
    ('Grade', 'A'),
    ('Status', True)
])

#dict 안에 key = name 형태로 나열할 수도 있음.
f = dict(
    Name = 'Niceman',
    City = 'Seoul',
    Age = 33,
    Grade = 'A',
    Status = True
)

f_2 = [d,e,f]  
print(f_2) 
'''
list에 넣어서 많이 사용함. (json 형태)
[{'Name': 'Niceman', 'City': 'Seoul', 'Age': 33, 'Grade': 'A', 'Status': True}, 
{'Name': 'Niceman', 'City': 'Seoul', 'Age': 33, 'Grade': 'A', 'Status': True}, 
{'Name': 'Niceman', 'City': 'Seoul', 'Age': 33, 'Grade': 'A', 'Status': True}]
'''

print('a - ', type(a), a)#a -  <class 'dict'> {'name': 'Kim', 'phone': '01033337777', 'birth': '870514'}
print('b - ', type(b), b)#b -  <class 'dict'> {0: 'Hello Python'}
print('c - ', type(c), c)#c -  <class 'dict'> {'arr': [1, 2, 3, 4]}
print('d - ', type(d), d)#d -  <class 'dict'> {'Name': 'Niceman', 'City': 'Seoul', 'Age': 33, 'Grade': 'A', 'Status': True}
print('e - ', type(e), e)#d -  <class 'dict'> {'Name': 'Niceman', 'City': 'Seoul', 'Age': 33, 'Grade': 'A', 'Status': True}
print('f - ', type(f), f)#d -  <class 'dict'> {'Name': 'Niceman', 'City': 'Seoul', 'Age': 33, 'Grade': 'A', 'Status': True}


#출력
print('a - ', a['name'])  #a -  Kim key가 없는 경우 에러남.
print('a - ', a.get('name')) #a -  Kim 동일한 호출방법 key가 없는 경우 None이 나옴.

print('b -', b[0]) #b - Hello Python
print('b - ', b.get(0)) #b -  Hello Python

print('f - ', f.get('City')) #f - Seoul

#딕셔너리 추가
a['address'] = 'seoul'
print('a - ', a) #a -  {'name': 'Kim', 'phone': '01033337777', 'birth': '870514', 'address': 'seoul'}
a['rank'] = [1,2,3] #리스트 추가도 가능
print('a - ', a) #a -  {'name': 'Kim', 'phone': '01033337777', 'birth': '870514', 'address': 'seoul', 'rank': [1, 2, 3]}

#딕셔너리 길이
print('a - ', len(a))


#dict_keys, dict_values, dict_items : 반복문(_iter_)에서 사용 가능
#  key값들만 가져오는 함수
print('a - ', a.keys()) #a -  dict_keys(['name', 'phone', 'birth', 'address', 'rank']) 
# key들 리스트로 가져옴
print('a - ', list(a.keys())) #a -  ['name', 'phone', 'birth', 'address', 'rank'] 
#value 들 가져옴
print('a - ', a.values()) #a -  dict_values(['Kim', '01033337777', '870514', 'seoul', [1, 2, 3]])
#value들 리스트로 가져옴
print('a - ', list(a.values())) #a -  ['Kim', '01033337777', '870514', 'seoul', [1, 2, 3]]
#key value 모두 가져옴
print('a - ', a.items()) #a -  dict_items([('name', 'Kim'), ('phone', '01033337777'), 
#('birth', '870514'), ('address', 'seoul'), ('rank', [1, 2, 3])])
#리스트로 key value 가져옴
print('a - ', list(a.items())) #a -  [('name', 'Kim'), ('phone', '01033337777'), 
#('birth', '870514'), ('address', 'seoul'), ('rank', [1, 2, 3])]
#pop() 값을 꺼내오고 딕셔너리에서 제거
print('a - ', a.pop('name'))#a -  Kim
print('a - ', a) #a -  {'phone': '01033337777', 'birth': '870514', 'address': 'seoul', 'rank': [1, 2, 3]}
#pop() 값을 꺼내오고 딕셔너리에서 제거
print('a - ', a.popitem()) #a -  ('rank', [1, 2, 3]
print('a -', a) #a - {'phone': '01033337777', 'birth': '870514', 'address': 'seoul'}
#키가 딕셔너리 안에 있는지 여부
print('a - ', 'birth' in a) #a -  True


#수정
a['address'] = 'deajeon'
print(a) #{'phone': '01033337777', 'birth': '870514', 'address': 'deajeon'}
#동일 역할
a.update(address = 'incheon')
print(a) #{'phone': '01033337777', 'birth': '870514', 'address': 'incheon'}
#동일 역할
temp = {'address' : 'busan'}
a.update(temp)
print(a) #{'phone': '01033337777', 'birth': '870514', 'address': 'busan'}




