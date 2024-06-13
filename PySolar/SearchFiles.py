import pysolr
import json


def searchMovies(solr):
    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query:")
        if command == '':
            break

        print()
        print("Searching for:", command)
        results = solr.search('title:' + command)
        print("%s total matching documents." % len(results))

        for result in results:
            print(result['id'], '\t', result['title'])


if __name__ == '__main__':
    solr_url = 'http://localhost:8983/solr/demo-core'
    solr = pysolr.Solr(solr_url)
    response = solr.ping()
    print("Solr Status: " + json.loads(response)['status'])

    searchMovies(solr)
