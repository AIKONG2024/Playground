import subprocess

# 반복 실행
img_sizes = [640, 1024, 1280]  
batch_sizes = [16, 32]  
epoch_numbers = [100, 200, 300] 
pretrained_w = "face_detection_yolov5s.pt"
datasets_yaml = "data/class_kor.yaml"
cfg_models = ["yolov5s.yaml"]



def run(img_sizes, batch_sizes, epoch_numbers):
    for img in img_sizes:
        for batch in batch_sizes:
            for epoch in epoch_numbers:
                for cft_model in cfg_models:
                    name = f"is{img}b{batch}e{epoch}cf{cft_model}"
                    command = f'''python train.py --img {img} --batch {batch} 
                    --epochs {epoch} --weights {pretrained_w} --data {datasets_yaml} 
                    --cfg{cft_model} --name {name}'''
                    subprocess.run(command, shell=True)
# 실행 함수를 호출
run(img_sizes, batch_sizes, epoch_numbers)

# ====
#yolov5
# import subprocess

# img_sizes = [640, 1024, 1280]
# batch_sizes = [16, 32]
# epoch_numbers = [100, 200, 300]
# pretrained_w = "face_detection_yolov5s.pt"
# datasets_yaml = "class_kor.yaml"
# cfg_models = ["yolov5s.yaml"]
# path = "train.py"


# def run(
#     img_sizes, batch_sizes, epoch_numbers, pretrained_w, datasets_yaml, cfg_models, path
# ):
#     for img in img_sizes:
#         for batch in batch_sizes:
#             for epoch in epoch_numbers:
#                 for cft_model in cfg_models:
#                     name = f"is_{img}_b_{batch}_e_{epoch}_cf_{cft_model}"
#                     command = f"python {path} --img {img} --batch {batch} --epochs {epoch} --weights {pretrained_w} --data {datasets_yaml} --cfg {cft_model} --name {name}"
#                     subprocess.run(command, shell=True, check=True, encoding="utf-8")


# run(img_sizes, batch_sizes, epoch_numbers, pretrained_w, datasets_yaml, cfg_models, path)
#test