from sklearn.preprocessing import OneHotEncoder, LabelEncoder

animal = [['개'],["말"], ["소"], ["고양이"], ["공룡"]]
print(animal)#[['개'], ['말'], ['소'], ['고양이'], ['공룡']]
onehot_e_animal = OneHotEncoder(sparse_output=False).fit_transform(animal)
print(onehot_e_animal)
'''
[[1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]]
'''


#LabelEncoder
animal = ['개',"말", "소", "고양이", "공룡"]

print(animal)#['개', '말', '소', '고양이', '공룡']
label_e_animal = LabelEncoder().fit_transform(animal)
print(label_e_animal)#[0 3 4 1 2]