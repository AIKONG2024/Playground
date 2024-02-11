import obesity03_train01_catb
import obesity03_train02_xgb
import obesity03_train03_lgbm
import obesity03_train04_rf

from random import randint

model_file_list= [obesity03_train01_catb ,obesity03_train02_xgb,
                  obesity03_train03_lgbm,obesity03_train04_rf]

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