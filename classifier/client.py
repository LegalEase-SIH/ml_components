import requests

data = {
      'petitionId' : "askjsdhf2342313jhkjh42",
      'url': "https://firebasestorage.googleapis.com/v0/b/legalease-bb0ad.appspot.com/o/resources%2F20BRS1161_DACA_Lab6.pdf%7D?alt=media&token=608d4132-69cc-4277-a213-ca537ae66628"  
}

url = 'http://127.0.0.1:8080/petitionSuccessProb'

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())