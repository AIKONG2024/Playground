#모듈사용 실습
import sys

print(sys.path) #이 경로 안에 모듈, 패키지가 담겨있음 영구적으로 등록된것 아님.


print(type(sys.path))

#모듈 경로 삽입
sys.path.append('/Users/kongseon-eui/Documents/math') #코드상에서 append로 가져다 쓸수 있음.

print(sys.path)
import test_module


print(test_module.power(10,3))
=



