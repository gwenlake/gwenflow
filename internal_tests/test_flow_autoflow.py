import context

from gwenflow import ChatOpenAI, AutoFlow
from gwenflow.tools import WikipediaTool, WebsiteReaderTool, DuckDuckGoSearchTool, DuckDuckGoNewsTool, PDFTool
from gwenflow import set_log_level_to_debug

set_log_level_to_debug()

# https://wellsfargo.bluematrix.com/links2/html/a4c7832f-4cd5-434c-926a-90eef3d92eb9
# https://www.vanguard.co.uk/content/dam/intl/europe/documents/en/vanguard-economic-and-market-outlook-for-2025-beyond-the-landing-gbp-en-pro.pdf
# https://mlaem.fs.ml.com/content/dam/ML/ecomm/pdf/Viewpoint_November_2024_Merrill.pdf
# https://think.ing.com/uploads/reports/Macro_Outlook_Dec_24_final.pdf
# https://s3-eu-west-1.amazonaws.com/euissmultisiteprod-live-8dd1b69cadf7409099ee6471b87c49a-7653963/international/PDF/download-material/outlook-2025.pdf
# https://www.deutschewealth.com/content/dam/deutschewealth/insights/investing-insights/economic-and-market-outlook/2025/PERSPECTIVES-Annual-Outlook-2025.pdf
# https://www.axa-im.com/sites/corporate/files/2024-12/2024%2012%2004%20Outlook%202025-v2.pdf
# https://docfinder.bnpparibas-am.com/api/files/ebe09a12-14ee-4c25-aff3-9c66d36157a1
# https://www.blackrock.com/corporate/literature/whitepaper/bii-global-outlook-2025.pdf
# https://privatebank.barclays.com/content/dam/privatebank-barclays-com/en-gb/private-bank/documents/insights/outlook-2025/outlook-2025.pdf
# https://www.goldmansachs.com/images/insights/2025-outlooks/Tailwinds-Probably-Trump-Tariffs.pdf

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

task = """Peux tu analyser les développements récents sur le Real brésilien, et notamment les liens entre la politique de la banque centrale, l'inflation et le taux de change."""

autoflow = AutoFlow(llm=llm, tools=[DuckDuckGoNewsTool(), DuckDuckGoSearchTool(), PDFTool(), WebsiteReaderTool()])
response = autoflow.run(task)
print(response["content"])