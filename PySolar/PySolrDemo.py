import IndexFiles
import SearchFiles

import pysolr
import json

if __name__ == '__main__':
    solr_url = 'http://localhost:8983/solr/demo-core'
    solr = pysolr.Solr(solr_url)
    response = solr.ping()
    print("Solr Status: " + json.loads(response)['status'])

    IndexFiles.indexMovies(solr)
    SearchFiles.searchMovies(solr)
