#함수의 장점
# 1. 반복되는 부분을 함수로 만들어서 사용하면 코드가 간결해진다.
# 2. 프로그램의 흐름을 일목요연하게 볼 수 있다.
# 3. 코드의 재사용성이 높다.
# 4. 코드의 수정이 용이하다.
# 5. 안정성이 좋아지고 개발 집중도가 증가한다.
# 6. 디버깅이 쉬워진다.

#함수의 선언
#파이썬 함수식 및 람다(Lambda)

# 함수 정의 방법
# def function_name(parameter):
#    code


# 예제1
def first_func(w):
    print("Hello, ", w)
    
word = "Goodboy"

first_func(word) #Hello, Goodboy
    
    
#예제2
#def : 정의 define 
def return_func(w1):
    value = "Hello, " + str(w1)
    return value

x = return_func('Goodboy2') 
print(x) #Hello, Goodboy2


#예제3(다중반환)

def func_mul(x):
    y1 = x * 10
    y2 = x * 20
    y3 = x * 30
    return y1, y2, y3

x, y, z = func_mul(10) #언팩킹

print(x,y,z) #100, 200, 300


#튜플 리턴
def func_mul2(x):
    y1 = x * 10
    y2 = x * 20
    y3 = x * 30
    return (y1, y2, y3)

q = func_mul2(20)
print(type(q), q, list(q))


#리스트 리턴
def func_mul2(x):
    y1 = x * 10
    y2 = x * 20
    y3 = x * 30
    return [y1, y2, y3]

p = func_mul2(30)

print(type(p), p, set(q))


#딕셔너리 리턴
def func_mul3(x):
    y1 = x * 10
    y2 = x * 20
    y3 = x * 30
    return {'v1':y1, 'v2': y2, 'v3': y3}

d = func_mul3(30)

print(type(d), d, d.get('v2'), d.items(), d.keys())

#중요
# *args, **kwargs


# positional arguments
# *args(언팩킹)
def args_func(*args): # 매개변수 명 자유
    for i, v in enumerate(args):
        print(f'Result : {i, v}')
    print('------')
    
args_func('Lee') #Result : (0, 'Lee')
args_func('Lee', 'Park')#Result : (0, 'Lee') Result : (1, 'Park')
args_func('Lee', 'Park', 'Kim')#Result : (0, 'Lee') Result : (1, 'Park') Result : (2, 'Kim')

# keyword arguments
# **kwarg(언팩킹)
def kwargs_func(**kwargs):
    for v in kwargs.keys():
        print("{}".format(v), kwargs[v])
    print('-----')

kwargs_func(name1 = 'Lee')
kwargs_func(name1 = 'Lee', name2 = 'Park')
kwargs_func(name1 = 'Lee', name2 = 'Park', name3 = 'Cho', sendSME = False)

# 전체 혼합
def example(args_1, args_2, *args, **kwargs):
    print(args_1, args_2, args, kwargs)


example(10, 20, 'Lee', 'Kim', 'Park', age1=20, age2=30, age3=40)




def train_test_split(*args, train_size = None, random_state = None):
    list = []
    test_x = []
    train_x = []
    
    for i,v in enumerate(args):
        if i < len(x) * train_size :
            train_x.append(v)
        test_x.append(v)
    list.append(train_x)
    list.append(test_x)
    return list
    
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
train_x, test_x = train_test_split(x,train_size=0.7,random_state=1)

print(train_x)
print(test_x)


#중첩함수

def nested_func(num):
    def func_in_func(num):
        print(num)
    print("In func")
    func_in_func(num + 100)

nested_func(100)


#람다식 예제
#메모리 절약, 가독성 향상, 코드 간결
# 함수는 객체 생성 -> 리소스(메모리) 할당
#람다는 즉시 실행 함수(Heap 초기화) -> 메모리 초기화
#남발 시 가독성 오히려 감소

def mul_func(x,y):
    return x * y


a = lambda x, y:x*y
print(a(5, 6))

def mul_func(x,y):
    return x * y
#일반적인 함수는 객체가 메모리()에 들어감.

q =mul_func(10,50)
print(q)

lambda_mul_func = lambda x,y:x*y
#람다는 변수에 할당되고 초기화(heap)
print(lambda_mul_func(10,50))

#heap 메모리는 동적으로 할당된 메모리 

#파이썬은 모든게 객체.
#값을 heap에 저장하고 stack에서 참조함.
#껍데기는 스텍, 알맹이는 heap 에 저장된다고 생가하면 되는건가여..

def func_final(x, y, func):
    print(x*y*func(100,100))
    
func_final(10, 20, lambda x,y:x*y)