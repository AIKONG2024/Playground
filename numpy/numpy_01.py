import numpy as np #가독성 numpy = np 많은 개발자들이 이렇게 사용

print(np.__version__) 
a = [[1,2,3], [2,3,4], [5,6,7]] 
print(a)#[[1, 2, 3], [2, 3, 4], [5, 6, 7]]
np_a = np.array(a) #배열을 행렬로 표시
print(np_a)
'''
[[1 2 3]
 [2 3 4]
 [5 6 7]]
'''
print(a[0][1]) #2
print(np_a[0][1]) #2
print(a[-1][-1]) #7
print(np_a[-1][-1]) #7
print(np_a[2,]) #[5 6 7]
print(np.sum(np_a)) #33 행렬의 모든 원소 더함 

#list comprebasion(루프)

comprebasion = np.array(range(i, i+3) for i in [1,4,7])
#1이 i로 전달됨 1,2,3 생성 -> 4가 i로 전달됨 4,5,6 생성 -> 7이 i로 전달됨 7,8,9
print(comprebasion.shape)

zeros_1 = np.zeros(5)
print(zeros_1) #[0. 0. 0. 0. 0.]

zeros_2 = np.zeros((3,5))
print(zeros_2)
'''
[0. 0. 0. 0. 0.]
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
'''

ones_1 = np.ones((3,5))
print(ones_1)
'''
[[1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]]
'''

full_1 = np.full((3,5),5)
print(full_1)
'''
[[5 5 5 5 5]
 [5 5 5 5 5]
 [5 5 5 5 5]]
'''

arange_1 = np.arange(0, 10, 2) #0~9까지 2씩 증가
print(arange_1) #[0 2 4 6 8]

linespace = np.linspace(0, 100, 5, dtype = int) #0~100 을 5개로 균등하게 나눔
print(linespace) #[  0  25  50  75 100]


rand_1 = np.random.random((3,3)) #0~1 사이의 랜덤 3x3
print(rand_1)
'''
[[0.06038789 0.82357611 0.81457934]
 [0.36328032 0.94497993 0.79900866]
 [0.57950431 0.34185231 0.57057225]]
'''

rand_2 = np.random.randint(0,10,(3,3)) #0~9까지 정수 3x3 랜덤
print(rand_2)
'''
[[3 1 8]
 [0 1 3]
 [1 9 7]]
'''

rand_3 = np.random.normal(0, 1, (1,10)) #정규분포 mu는 평균, sigma는 표준편차 
#0을 기준으로 왼쪽 -, 오른쪽 + 값 정규분포처럼 가운데 불룩한 모양의 랜덤값 생성
print(rand_3)
#[[ 1.78411429  0.88286863 -0.60189759  0.58298902  0.40427558  0.20610827 -0.46717522 -0.99757422 -0.28784226  0.01615689]]

np.random.seed(0)
arr1 = np.random.randint(10, size= 6) # 시드에 따라 랜덤값이 변경. 시드가 동이하면 랜덤값도 동일.
arr2 = np.random.randint(10, size= (2,3)) 
print(arr1) #[5 0 3 3 7 9]
print(arr2)
'''
[[3 5 2]
 [4 7 6]]
'''
print(f'차원 {arr1.ndim}, 구조 : {arr1.shape}, size : {arr1.size}, 데이터타입: {arr1.dtype}') #차원 1, 구조 : (6,), size : 6, 데이터타입: int64
print(f'차원 {arr2.ndim}, 구조 : {arr2.shape}, size : {arr2.size}, 데이터타입: {arr2.dtype}') #차원 2, 구조 : (2, 3), size : 6, 데이터타입: int64

#인덱싱 : 단일 원소 접근
print(arr1[0]) #5
print(arr1[-1]) #9 가장 마지막 원소 
print(arr1[-3]) #3
print(arr2[-1][-2]) #7
print(arr2[-1]) #[4 7 6]

#수정
arr2[-1,-2] = 77
print(arr2)
'''
[[ 3  5  2]
 [ 4 77  6]]
'''
arr2[0] = [1,2,3]
print(arr2)
'''
[[ 1  2  3]
 [ 4 77  6]]
'''

#슬라이싱
#[start:end:step] end는 exclusive 불포함
arange_2 = np.arange(10)
print(arange_2) #[0 1 2 3 4 5 6 7 8 9]
print(arange_2[0:8:1] ) #[0 1 2 3 4 5 6 7]
print(arange_2[:5:1] )# [0 1 2 3 4]
print(arange_2[:4:])# [0 1 2 3]
print(arange_2[:2] )#[0 1] #모든게 생략 가능해도 end는 무조건 포함 되어야함.
print(arange_2[:]) #[0 1 2 3 4 5 6 7 8 9]
print(arange_2[::])#[0 1 2 3 4 5 6 7 8 9]
print(arange_2[2:9:2]) #[2 4 6 8]
print(arange_2[2::2]) # [2 4 6 8]
print(arange_2[::-1])# 거꾸로 출력 [9 8 7 6 5 4 3 2 1 0]
print(arange_2[-1:-11:-1])#[9 8 7 6 5 4 3 2 1 0] -1=9, -11=범위 밖이므로 0, -1 거꾸로 
print(arange_2[5::-1]) #[5 4 3 2 1 0] 1.거꾸로를 기준으로 잡음. 시작값: 5 마지막값 : 0

#형태변환
arange_3 = np.arange(12)
print(arange_3.reshape(-1,4)) #4개씩 잘라줌
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
arange_4 = np.arange(7)
arange_5 = np.array((7,8,9,10))
print(arange_4, arange_5) #[0 1 2 3 4 5 6] [ 7  8  9 10]
#두 배열을 합침
print(np.concatenate([arange_4, arange_5])) #[ 0  1  2  3  4  5  6  7  8  9 10]
print(np.concatenate([arange_4, arange_5], axis=0)) #[ 0  1  2  3  4  5  6  7  8  9 10]

arr_5 = np.array([[1,2,3],[4,5,6]])
arr_6 = np.array([[1,2,3],[4,5,6]])
print(np.concatenate([arr_5, arr_6], axis= 0))
'''
[[1 2 3]
 [4 5 6]
 [1 2 3]
 [4 5 6]]
 '''
print(np.concatenate([arr_5, arr_6], axis= 1))
'''
[[1 2 3 1 2 3]
 [4 5 6 4 5 6]]
''' 

print(np.vstack([arr_5, arr_6]))
'''
[[1 2 3]
 [4 5 6]
 [1 2 3]
 [4 5 6]]
'''
print(np.hstack([arr_5, arr_6]))
'''
[[1 2 3 1 2 3]
 [4 5 6 4 5 6]]
'''



