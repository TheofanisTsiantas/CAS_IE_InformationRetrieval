INDEX_DIR = "IndexFiles.index"

import os, lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import NIOFSDirectory
from org.apache.lucene.search import IndexSearcher

# Package for displaying texted wrapped in the window
import textwrap


def search_movies(analyzer):
    # Configuration settings
    directory = NIOFSDirectory(Paths.get(os.path.join(os. getcwd(), INDEX_DIR)))
    searcher = IndexSearcher(DirectoryReader.open(directory))
    # Loop for queries
    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query:")
        if command == '':
            break

        print()
        print("Searching for:", command)
        query = QueryParser("plot", analyzer).parse(command)
        score_docs = searcher.search(query, 50).scoreDocs
        print("%s total matching documents." % len(score_docs))

        # id of each displayed plot
        plot_id = 0
        for scoreDoc in score_docs:
            plot_id += 1
            doc = searcher.doc(scoreDoc.doc)
            print(str(plot_id), '\t', doc.get("year"),  '\t', doc.get("title"), '\t', '\t--> Score:', scoreDoc.score)

        # If at least a plot was found then plot the title and prompt the user to see the plot
        if plot_id > 0:
            while True:
                print()
                print("Type the number of the movie you want to see the plot for.")
                print("Type anything else to return to the query search.")
                inp = input("Movie number: ")
                if inp.isdigit():
                    parsed_number = int(inp)
                    if 0 < parsed_number < len(score_docs):
                        doc = searcher.doc(score_docs[parsed_number].doc)
                        plot = doc.get("plot")
                        if plot == "":
                            print("Sorry, this movie has no plot available")
                        else:
                            wrapped_plot = textwrap.fill(plot, width=50)
                            print(wrapped_plot)
                else:
                    break
    del searcher


if __name__ == '__main__':
    # Start of PyLucene VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    # Start the query method
    search_movies(StandardAnalyzer())