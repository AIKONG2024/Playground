import obesity03_train01_catb_v2
import obesity03_train02_xgb_v2
import obesity03_train03_lgbm_v2
import obesity03_train04_rf_v2

from random import randint

model_file_list= [obesity03_train01_catb_v2 ,obesity03_train02_xgb_v2,
                  obesity03_train03_lgbm_v2,obesity03_train04_rf_v2]
def do():
    while 1 :
        rand_idx = randint(0,3)
        model_file = model_file_list[rand_idx]
        print("================================")
        print(model_file.__name__ + "START")
        print("================================")
        
        model_file.obtuna_tune()

def main():
    do()
    # GridSearchCV_tune()

if __name__ == '__main__':
    main()