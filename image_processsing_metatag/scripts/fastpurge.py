
import requests
from akamai.edgegrid import EdgeGridAuth
from urllib.parse import urljoin

# Akamai credentials and base URL
baseurl = 'https://akab-rftchmwemn5pfikw-ntyuihphbvzqdxn3.luna.akamaiapis.net/'  # Host value from your credentials
s = requests.Session()
s.auth = EdgeGridAuth(
    client_token='akab-mu2gxcltb4h7pzbd-llmhjltbbj2rsmkv',
    client_secret='z7LR/XG78NLESxM+Ob9v8o0ltIWcbH5o7ENRxCGgrN8=',
    access_token='akab-sr2msln7pkdglpxi-wyrjgmteoaoya5kl'
)


# Define the API endpoint and payload
endpoint = '/ccu/v3/delete/url'
url = urljoin(baseurl, endpoint)
payload = {
    "objects": [
        "https://content1.jdmagicbox.com/test_img_upload/output/test_data_1.jpg",
        "https://content2.jdmagicbox.com/test_img_upload/output/test_data_1.jpg",
        "https://content3.jdmagicbox.com/test_img_upload/output/test_data_1.jpg",
        "https://content4.jdmagicbox.com/test_img_upload/output/test_data_1.jpg",
        "https://images.jdmagicbox.com/test_img_upload/output/test_data_1.jpg"
    ]
}

# Send the POST request
response = s.post(url, json=payload)

print(response)


# Print the response
print("Status Code:", response.status_code)
if response.status_code == 201:
    print("Success:", response.json())
else:
    print("Error:", response.text)


# import requests
# from akamai.edgegrid import EdgeGridAuth
# from urllib.parse import urljoin

# baseurl = 'https://akab-rftchmwemn5pfikw-ntyuihphbvzqdxn3.luna.akamaiapis.net/' # this is the "host" value from your credentials file
# s = requests.Session()
# s.auth = EdgeGridAuth(
#     client_token='akab-mu2gxcltb4h7pzbd-llmhjltbbj2rsmkv',
#     client_secret='z7LR/XG78NLESxM+Ob9v8o0ltIWcbH5o7ENRxCGgrN8=',
#     access_token='akab-sr2msln7pkdglpxi-wyrjgmteoaoya5kl'
# )

# result = s.get(urljoin(baseurl, '/ccu/v3/delete/url'))
# print(result.status_code)
# print(result.json())
