import httpx

url = "https://gwenlake-public.s3.amazonaws.com/SR15_Summary_Volume_french.pdf"
response = httpx.get(url)
print(response)
