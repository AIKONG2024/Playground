import numpy as np
a = np.array([-192222222222, -1000, -1, 0, 1,10, 100, 1000000, 1000000000]).reshape(-1,1) 


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(a)
print(standard_scaler.transform(a))
'''
[[-2.82838951]
 [ 0.35147877]
 [ 0.35147879]
 [ 0.35147879]
 [ 0.35147879]
 [ 0.35147879]
 [ 0.35147879]
 [ 0.35149533]
 [ 0.36802146]]
'''

import numpy as np
a = np.array([-192222222222, -1000, -1, 0, 1,10, 100, 1000000, 1000000000]).reshape(-1,1) 

from sklearn.preprocessing import MinMaxScaler
standard_scaler = MinMaxScaler()
standard_scaler.fit(a)
print(standard_scaler.transform(a))
'''
[[0.        ]
 [0.99482461]
 [0.99482461]
 [0.99482461]
 [0.99482461]
 [0.99482461]
 [0.99482461]
 [0.99482979]
 [1.        ]]
'''

import numpy as np
a = np.array([-192222222222, -1000, -1, 0, 1,10, 100, 1000000, 1000000000]).reshape(-1,1) 

from sklearn.preprocessing import MaxAbsScaler
standard_scaler = MaxAbsScaler()
standard_scaler.fit(a)
print(standard_scaler.transform(a))
'''
[[-1.00000000e+00]
 [-5.20231214e-09]
 [-5.20231214e-12]
 [ 0.00000000e+00]
 [ 5.20231214e-12]
 [ 5.20231214e-11]
 [ 5.20231214e-10]
 [ 5.20231214e-06]
 [ 5.20231214e-03]]
'''

import numpy as np
a = np.array([-192222222222, -1000, -1, 0, 1,10, 100, 1000000, 1000000000]).reshape(-1,1) 

from sklearn.preprocessing import RobustScaler
standard_scaler = RobustScaler()
standard_scaler.fit(a)
print(standard_scaler.transform(a))
'''
[[-1.90319032e+09]
 [-9.91089109e+00]
 [-1.98019802e-02]
 [-9.90099010e-03]
 [ 0.00000000e+00]
 [ 8.91089109e-02]
 [ 9.80198020e-01]
 [ 9.90098020e+03]
 [ 9.90099009e+06]]
'''