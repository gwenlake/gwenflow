import context
import json

from gwenflow import set_log_level_to_debug
from gwenflow import SimpleDirectoryReader
from gwenflow.readers import PDFReader, JSONReader, TextReader, WebsiteReader

set_log_level_to_debug()

reader = TextReader()
documents = reader.read("./documents/test.txt")
print(json.dumps(documents[0].model_dump(), indent=4))

# reader = PDFReader()
# documents = reader.read("./documents/sample.pdf")
# print(json.dumps(documents[0].model_dump(), indent=4))

reader = PDFReader()
documents = reader.read("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
print(json.dumps(documents[0].model_dump(), indent=4))


# reader = JSONReader()
# documents = reader.read("./documents/test/transport-securise-des-batteries-au-lithium-tout-c.json")
# print(json.dumps(documents[0].model_dump(), indent=4))

# reader = WebsiteReader(delay=False)
# documents = reader.read("https://gwenlake.com")
# for d in documents:
#     print(json.dumps(d.model_dump(), indent=4))

# BUCKET = "glk-datasets-001"
# FILE   = "ffe5fb2e-c4b0-4402-89f5-cf842fa62082/files/2024/12/_why_do_we_need_to_strengthen_climate_adaptations_scenarios_and_financial_lines_of_defence.pdf"

# pdfreader = PDFReader()
# documents = pdfreader.read(f"s3://{BUCKET}/{FILE}")
# print(documents)


# reader = SimpleDirectoryReader(input_dir="./documents")
# documents = reader.read(show_progress=True)

# print(documents[0])
# print(documents[3])

# reader = SimpleDirectoryReader(input_dir="./documents", required_exts=[".pdf"])
# documents = reader.read(show_progress=True)
# print(documents[0])

# reader = SimpleDirectoryReader(input_dir="./documents")
# documents = reader.read(show_progress=True)

# reader = SimpleDirectoryReader(input_dir="./documents", required_exts=[".csv"])
# documents = reader.read(show_progress=True)
