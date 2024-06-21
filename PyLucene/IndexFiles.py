INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
import subprocess
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import NIOFSDirectory


# Function for installing a package during run time
def install(package):
    print("inside install")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install pandas to manipulate the movies csv
install('pandas')
import pandas as pd


def indexMovies(analyzer, movies: pd.core.frame.DataFrame):
    # Configuration settings
    store = NIOFSDirectory(Paths.get(os.path.join(os. getcwd(), INDEX_DIR)))
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(store, config)

    t1 = FieldType()
    t1.setStored(True)
    t1.setTokenized(True)
    t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    # Add movies to documents for indexing
    idx = 0
    indexed_movies = []
    print("Reading movies")
    for index, row in data.iterrows():
        doc = Document()
        doc.add(Field("year", row['Release Year'], t1))
        doc.add(Field("title", row['Title'], t1))
        doc.add(Field("plot", row['Plot'], t1))
        indexed_movies.append(doc)
        idx += 1

    # Info print
    print("Indexing "+str(idx)+" movies")
    # Index documents (movies)
    start_index_time = time.time()
    idx = 0
    for doc in indexed_movies:
        if idx % 1000 == 0: print(".", end='', flush=True)
        try:
            writer.addDocument(doc)
            idx += 1
        except Exception as e:
            print("Failed in indexMovies:", e)
    elapsed_time = round(time.time() - start_index_time,1)
    print("Successfully indexed "+str(idx)+" movies")
    print(f"Elapsed time (indexing): {elapsed_time} seconds")
    writer.commit()
    writer.close()
    print("Successfully committed the movies")


if __name__ == '__main__':
    # Start of PyLucene VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    # Read and index the movies
    data = pd.read_csv('../Movie_Plots.csv')
    indexMovies(StandardAnalyzer(), data)
