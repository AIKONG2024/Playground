import obesity03_train01_catb
import obesity03_train02_xgb
import obesity03_train03_lgbm
import obesity03_train04_rf
from random import randint

model_file_list= [obesity03_train01_catb.obtuna_tune() ,obesity03_train02_xgb.obtuna_tune(),
                  obesity03_train03_lgbm.obtuna_tune(),obesity03_train04_rf.obtuna_tune()]

while 1 :
    model_file_list[randint(0,3,1)]