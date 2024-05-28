import requests

url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
output = "C:\\stylegan2-ffhq-config-f.pkl"

response = requests.get(url)
with open(output, 'wb') as file:
    file.write(response.content)
    print("완료")
