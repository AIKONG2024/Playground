import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 모델과 토크나이저 불러오기
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 모델을 평가 모드로 설정
model.eval()

# 텍스트 생성 함수
def generate_text(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # GPU가 사용 가능한 경우 GPU로 모델을 이동
    if torch.cuda.is_available():
        model.to('cuda')
        input_ids = input_ids.to('cuda')

    # 텍스트 생성
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, pad_token_id=tokenizer.eos_token_id)
    
    # 결과 디코딩 및 출력
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_texts

# 예제 사용
prompt = "너 누구야? 팍씨"
generated_text = generate_text(prompt, max_length=100, num_return_sequences=1)
print(generated_text[0])