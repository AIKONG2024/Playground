#클래스를 사용하므로서
#코드의 재사용성이 좋다.
#개선 수정 버그 발생시 유지보수,
#주변에 영향(사이드이펙트) 이 나왔을때 최소화 (경제적)
#파이썬은 객체지향언어 (OOP : Procedual Oriented Programming)

#Self, 인스턴스 메소드, 인스턴스 변수

#클래스 and 인스턴스 차이 이해
#클래스변수: 직접 접근 가능, 공유
#인스턴스 변수: 객체마다 별도 존재

#예제1

class Dog: # object 상속 class Dog() 도 가능, class Dog(object) 도 가능
    #클래스 속성
    species = 'firstdog'
    
    #초기화/인스턴스 속성
    def __init__(self, name, age):
        self.name = name
        self.age = age
    #인스턴스와 객체 : 여기서 객체: Dog 인스턴스 : Dog()로 생성된 객체. 코드에 선언을 해서 메모리에 올라간 시점
        
#클래스 정보
print(Dog)

#인스턴스화
a = Dog("mikky", 2)
b = Dog('dage',3)

#비교
print(a == b, id(a), id(b)) #id 메모리 주소를 보는것.

#네임스페이스 : 객체를 인스턴스화 할떄 저장된 공간
print('dog1', a.__dict__) #__dict__내용을 알수 있음
print('dog1', b.__dict__)

# 인스턴스 속성 확인
print('{} is {} and {} is {}'.format(a.name, a.age, b.name, b.age))

if a.species == 'firstdog':
    print('{0} is a {1}'.format(a.name, a.species))
print(Dog.species)

#클래스 : Dog, 인스턴스 : Dog()


#예제2
#self의 이해
class SelfTest:
    def func1():
        print('Func1 called')
    def func2(self):
        print('Func2 called')
        print(id(self))


f = SelfTest()


print(dir(f)) #모든 사용 가능한 속성을 볼 수 있음.
print(id(f)) #메모리 주소
# f.func1() #예외
f.func2()

SelfTest.func1() #클래스로 바로 접근
SelfTest.func2(f) #에러남 인스턴스한 값을 넘겨주면 에러 안남.

#메서드 안에 self가 없으면 클래스 메서드네 하고 인스턴스 하지 않고 사용하면 되고, self가 있으면 인스턴스화 해서 사용해야함.


#예제 3
#클래스 변수, 인스턴스 변수
class Warehouse:
    # 클래스 변수
    stock_num = 0 # 재고
    
    #초기화할때 호출
    def __init__(self, name):
        #인스턴스 변수 : self 를 붙인것.
        self.name = name
        Warehouse.stock_num += 1
        
    #소멸할때 호출
    def __del__(self):
        Warehouse.stock_num -= 1
        
        
        
user1 = Warehouse('Lee') #stock_num = 1
user2 = Warehouse('Cho') #stock_num = 2

print(Warehouse.stock_num)
Warehouse.stock_num = 50 #직접접근 막아야함.
print(user1.name)
print(user2.name)
print(user1.__dict__)
print(user2.__dict__)
print('befor', Warehouse.__dict__)
print('>>>>', Warehouse.__dict__)

del user1
print('after', Warehouse.__dict__)


class Dog2: # object 상속 class Dog() 도 가능, class Dog(object) 도 가능
    #클래스 속성
    species = 'firstdog'
    
    #초기화/인스턴스 속성
    def __init__(self, name, age):
        self.name = name
        self.age = age
    #인스턴스와 객체 : 여기서 객체: Dog 인스턴스 : Dog()로 생성된 객체. 코드에 선언을 해서 메모리에 올라간 시점
    
    def info(self):
         return '{} is {} years old'.format(self.name, self.age)
    def speak(self,sound):
         return "{} says {}!".format(self.name, sound)
     
     

#인스턴스 생성
c = Dog2('july', 4)
d = Dog2('Marry', 10)
#메소드호출
print(c.info())
print(c.speak('Wal Wal'))
print(d.speak('Mung'))


