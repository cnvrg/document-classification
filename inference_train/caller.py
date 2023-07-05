import http.client
import json
import base64
conn = http.client.HTTPSConnection("inference-452-1.afbgghr4yqttu4sypilwvv.cloud.cnvrg.io", 443)

with open("/home/sp-01/Downloads/pdf4ofir1.pdf", "rb") as f:
    content = f.read()
    encoded1 = base64.b64encode(content).decode("utf-8")

rd = {'context':encoded1,"labels":['business', 'art & culture', 'politics']}

payload = '{"input_params":' + json.dumps(rd) + "}"
headers = {
    'Cnvrg-Api-Key': "wD1RPANERrfJCywhwaapCRFp",
    'Content-Type': "application/json"
    }

conn.request("POST", "/api/v1/endpoints/vhjq18wems9xdz3fx7aq", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
