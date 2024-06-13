import pysolr
import json

def indexMovies(solr):
    doc1 = {
        "id": "doc_1",
        "title": "Harry Potter and the Prisoner of Azkaban",
    }
    doc2 = {
        "id": "doc_2",
        "title": "Lord of the Rings: The fellowship of the ring",
    }
    doc3 = {
        "id": "doc_3",
        "title": "Toy Story 3",
    }

    print("Going to index 3 movies.")

    solr.add([
        doc1,
        doc2,
        doc3
    ])

    print("Movies Indexed Successfully!")


if __name__ == '__main__':
    solr_url = 'http://localhost:8983/solr/demo-core'
    solr = pysolr.Solr(solr_url)
    response = solr.ping()
    print("Solr Status: " + json.loads(response)['status'])

    indexMovies(solr)
