def decorator(func):
    def wrapper(*args, **kwargs):
        print("발표시작")
        result = func(*args, **kwargs)
        print("발표끝")
        return result
    return wrapper

@decorator
def test():
    print("발표 중 !!")
    
test()