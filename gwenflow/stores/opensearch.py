from typing import Any
import logging
import boto3
from opensearchpy import OpenSearch, helpers, AWSV4SignerAuth, RequestsHttpConnection

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


logger = logging.getLogger(__name__)


class OpenSearchDocumentStore():

    def __init__(self, uri, index, use_ssl=True, verify_certs=False, ca_certs=None, timeout=30, aws=False):
        if aws:
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, "eu-west-3", "es")
            self._client = OpenSearch(
                hosts = [{'host': uri, "port": 443}],
                http_auth = auth,
                use_ssl = True,
                verify_certs = True,
                connection_class = RequestsHttpConnection,
                pool_maxsize = 20
            )
        else:
            self._client = OpenSearch(uri, http_compress=True, use_ssl=use_ssl, verify_certs=verify_certs, ca_certs=ca_certs, ssl_assert_hostname=False, ssl_show_warn=False)
        self._index = index
        self._timeout = timeout

    @property
    def client(self) -> Any:
        return self._client

    @property
    def index(self) -> str:
        return self._index

    def count(self, query: str):
        q = { "query": query, "track_total_hits": True, "size": 1 }
        resp = self._client.search(body=q, index=self._index, request_timeout=self._timeout)
        return resp['hits']['total']['value']

    def delete(self, ids=None):
        if not ids:
            return self._client.indices.delete(index=self._index)
        if not isinstance(ids, list):
            ids = [ids]
        if len(ids)==0:
            logger.warning("No documents to be deleted from OpenSearch.")
            return False
        for id in ids:
            self._client.delete(index=self._index, id=id)
        return True

    def get(self, id: str):
        try:
            data = self._client.get(id=id, index=self._index)
            if "_source" in data:
                return data["_source"]
        except Exception as e:
            logger.error(repr(e))
        return None

    def put(self, documents, column_id="id"):
        if not isinstance(documents, list):
            documents = [documents]
        if len(documents)==0:
            logger.warning("No documents to be added to OpenSearch.")
            return False
        def _generator(documents):
            for doc in documents:
                yield {"_index" : self._index, "_id": doc[column_id], "_source": doc}
        success, failed = helpers.bulk(self._client, _generator(documents))
        if success < len(documents):
            logger.warning(f"Some documents failed to be added to OpenSearch. Failures: {failed}")
        logger.info(f"Indexed {success} documents")
        return True

    def search(self, query, aggs=None, sort=None, page=1, per_page=5000, fields=None, all=False, track_total_hits=True):

        # if all==True:
        #     per_page = 10000

        body = { "query": query, "size": per_page }

        if (page > 1) and (all == False):
            body["from"] = (page-1) * per_page

        if track_total_hits:
            body["track_total_hits"] = True

        if fields:
            body["fields"] = fields

        if aggs:
            body["aggs"]=aggs

        if sort:
            body["sort"] = sort

        # réfléchir au search after pour remplacer le scroll
        # if all:
        #     q["search_after"] = search_after
        
        documents = []
        aggregations = []
        if not all:
            resp = self._client.search(body=body, index=self._index, request_timeout=self._timeout)
            total = resp['hits']['total']['value']
            documents = [hit["_source"] for hit in resp['hits']['hits']]
            if aggs:
                aggregations = resp['aggregations']
            return {"object": "list", "total": total, "page": page, "per_page": per_page, "data": documents, "aggs":aggregations}

        resp = self._client.search(body=body, index=self._index, request_timeout=self._timeout, scroll='1m')
        for doc in resp['hits']['hits']:
            documents.append(doc["_source"])
        scroll_id = resp['_scroll_id']
        while len(resp['hits']['hits']):
            try:
                resp = self._client.scroll(scroll_id=scroll_id, scroll='5m')
                for doc in resp['hits']['hits']:
                    documents.append(doc["_source"])
                if scroll_id != resp["_scroll_id"]:
                    self._client.clear_scroll(scroll_id=scroll_id)
                    scroll_id = resp['_scroll_id']
            except Exception as e:
                logger.error(repr(e))
                break
        self._client.clear_scroll(scroll_id=scroll_id)
        total = len(documents)
        per_page = len(documents)
        return {"object": "list", "total": total, "page": page, "per_page": per_page, "data": documents, "aggs":aggregations}
