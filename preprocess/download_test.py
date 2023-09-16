import requests

response = requests.get('https://tkstock.site/wp-content/uploads/2021/09/line-delete-account-00.jpg') 

with open(f'sample.jpg', 'wb') as saveFile: 

    saveFile.write(response.content)