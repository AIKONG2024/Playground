# python 오류의 종류
# https://docs.python.org/ko/3.6/library/exceptions.html#KeyboardInterrupt

'''
예외처리
1. 오류 발생 재현
'''
#0
def zeroDivisionErrorInterrupt():
    4/0
#1    
def indexErrorInterrupt():
    a = [1,2,3]
    a[4]
    try:
        zeroDivisionErrorInterrupt()
        indexErrorInterrupt()
    except ZeroDivisionError as e:
        print(e) #division by zero
    except IndexError as e:
        print(e)

indexErrorInterrupt()