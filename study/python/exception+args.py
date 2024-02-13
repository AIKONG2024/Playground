# 에러 다루기 +  with args , kwargs

#error case
def do_index_error():
    ['a','b','c'][50]
def do_value_error():
    ['a','b','c'].index('z')
def do_zero_division_error():
    1/0
def do_type_error():
    1+"abc"
    
#error_message
error_messages = {
    "value_error": "참조값이 존재하지 않습니다.",
    "index_error": "Index를 찾을 수 없습니다.",
    "zero_division_error": "0을 나누어줄 수 없습니다.",
    "type_error": "잘못된 타입이 사용되었습니다.",
}

error_cases = do_value_error,do_index_error, do_zero_division_error, do_type_error

    
#tuple
# *args 튜플형태 - 몇개를 입력받을 지 모를때 사용, 언팩킹 1번
# *kwarg 딕셔너리형태 - 몇개를 입력받을 지 모를때 사용, 언팩킹 2번
def interrunt_error(*errors , **messages):
    
    for error in errors:
        try :  
            error()
        except ValueError as e: 
             print(f"[{e}] :" , messages.get("value_error"))
        except IndexError as e: 
             print(f"[{e}] :" , messages.get("index_error"))
        except ZeroDivisionError as e:
             print(f"[{e}] :" , messages.get("zero_division_error"))      
        except TypeError as e:
             print(f"[{e}] :" , messages.get("type_error"))
        finally:
            print("=========")

interrunt_error(*error_cases, **error_messages)
'''
결과:
['z' is not in list] : 참조값이 존재하지 않습니다.
=========
[list index out of range] : Index를 찾을 수 없습니다.
=========
[division by zero] : 0을 나누어줄 수 없습니다.
=========
[unsupported operand type(s) for +: 'int' and 'str'] : 잘못된 타입이 사용되었습니다.
=========
'''