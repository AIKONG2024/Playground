import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_text(prompt, max_length=1000, num_return_sequences=1, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    if torch.cuda.is_available():
        model.to('cuda')
        input_ids = input_ids.to('cuda')

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_length=max_length + len(input_ids[0]),
            num_return_sequences=num_return_sequences, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,           
            top_k=top_k,             
            top_p=top_p              
        )
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]    
    generated_responses = [text[len(prompt):].strip() for text in generated_texts]
    return generated_responses

# 예제 사용
prompt = "who is Sam Altman?"
generated_text = generate_text(prompt, max_length=1000, num_return_sequences=1)
print("프롬프트:\n", prompt)
print("===================================================")
print("생성 결과\n", generated_text[0])