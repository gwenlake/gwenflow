import context

from gwenflow.tools import WebsiteReaderTool


tool = WebsiteReaderTool()

# response = tool.run(url="https://www.abc.net.au/news/2025-01-02/south-korea-jeju-plane-crash/104776296")
# response = tool.run(url="https://www.reuters.com/world/asia-pacific/south-korea-police-raid-jeju-air-muan-airport-over-fatal-plane-crash-2025-01-02/")
response = tool.run(url="https://www.cbc.ca/news/world/south-korea-crash-jeju-1.7421344")
print(response)
