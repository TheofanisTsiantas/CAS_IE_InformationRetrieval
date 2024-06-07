#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import NIOFSDirectory
from org.apache.lucene.search import IndexSearcher

import textwrap

def searchMovies(analyzer):
    directory = NIOFSDirectory(Paths.get(os.path.join(os. getcwd(), INDEX_DIR)))
    searcher = IndexSearcher(DirectoryReader.open(directory))

    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query:")
        if command == '':
            break

        print()
        print("Searching for:", command)
        query = QueryParser("plot", analyzer).parse(command)
        scoreDocs = searcher.search(query, 50).scoreDocs
        print("%s total matching documents." % len(scoreDocs))
        id = 0;

        for scoreDoc in scoreDocs:
            id += 1
            doc = searcher.doc(scoreDoc.doc)
            print(str(id), '\t', doc.get("year"),  '\t', doc.get("title"), '\t', '\t--> Score:', scoreDoc.score)

        if id>0:
            while (True):
                print()
                print("Type the number of the movie you want to see the plot for.")
                print("Type anything else to return to the query search.")
                inp = input("Movie number: ")
                if inp.isdigit():
                    parsed_number = int(inp)
                    if 0 < parsed_number < len(scoreDocs):
                        doc = searcher.doc(scoreDocs[parsed_number].doc)
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
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)

    searchMovies(StandardAnalyzer())