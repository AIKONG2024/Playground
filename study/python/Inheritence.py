#상속
class 인공지능:
    type = "IT"
    def coding(self):
        print(f"type is {self.type}")
        print("do studing")

class Python:
    def ptn(self):
        print("do python")

class 머신러닝(인공지능, Python):
    type = "AI"
    def coding(self):
        print(f"type is {self.type}")
        print("do preprocessing")
    
class 딥러닝(머신러닝):
    def coding(self):
        print(f"type is {self.type}")
        print("do modeling")
AI = 인공지능()
ML = 머신러닝()
NN = 딥러닝()
AI.coding()
# type is IT
# do studing
ML.coding()
# type is AI
# do preprocessing
ML.ptn()
# do python
NN.coding()
# type is AI
# do modeling

