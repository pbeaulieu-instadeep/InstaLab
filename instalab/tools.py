"""This module provides the `ArxivSearch` class.

It is designed to interact with the
arXiv API (https://arxiv.org/). It allows users to search for academic papers
by keywords and retrieve the full text of specific papers.

The module utilizes the `arxiv` library for API communication and `pypdf`
for extracting text from downloaded PDF documents.
"""

from __future__ import annotations

import os
import time

import arxiv
from pypdf import PdfReader

MAX_QUERY_LENGTH = 300


class ArxivSearch:
    """A class to search for and retrieve academic papers from the arXiv repository.

    This class provides methods to find papers based on a query string and to
    download and extract the full text content of a specific paper by its ID.
    It handles API interactions, query processing, and PDF text extraction.

    Attributes:
        sch_engine (arxiv.Client): An instance of the arxiv API client.
    """

    def __init__(self):
        """Initializes the ArxivSearch class by creating an arxiv API client instance."""
        # Construct the default API client.
        self.sch_engine = arxiv.Client()

    def _process_query(self, query: str) -> str:
        """Processes a query string to fit within a predefined maximum length.

        If the query exceeds MAX_QUERY_LENGTH (300 characters), it truncates
        the query by whole words to stay within the limit, preserving as much
        of the initial query as possible.

        Args:
            query (str): The input search query string.

        Returns:
            str: The processed query string, potentially truncated.
        """
        if len(query) <= MAX_QUERY_LENGTH:
            return query

        # Split into words
        words = query.split()
        processed_query: list[str] = []
        current_length = 0

        # Add words while staying under the limit
        # Account for spaces between words
        for word in words:
            # +1 for the space that will be added between words
            if current_length + len(word) + (1 if processed_query else 0) <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + (1 if len(processed_query) > 1 else 0)
            else:
                break

        return " ".join(processed_query)

    def find_papers_by_str(self, query: str, n: int = 20) -> str | None:
        """Searches arXiv for papers matching a given query string and returns summaries.

        The query is first processed to adhere to length constraints. The search
        is performed against the abstracts of papers, sorted by relevance.
        If the API request fails, it retries up to a defined maximum number of times.

        Args:
            query (str): The search query string (e.g., "quantum computing").
            n (int, optional): The maximum number of paper summaries to retrieve.
                               Defaults to 20.

        Returns:
            str | None: A string containing concatenated summaries of the found
                        papers (Title, Summary, Publication Date, arXiv ID),
                        separated by newlines. Returns None if the search fails
                        after multiple retries or if no papers are found.
        """
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=n,
                    sort_by=arxiv.SortCriterion.Relevance,
                )

                paper_sums = []
                # `results` is a generator; you can iterate over its elements one by one...
                results_iterator = self.sch_engine.results(search)
                for r in results_iterator:
                    paperid = r.get_short_id()  # More robust way to get ID
                    pubdate = str(r.published.date())  # Get only date part
                    newline_char = "\n"
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary.replace(newline_char, ' ')}\n"  # Replace newlines in summary
                    paper_sum += f"Publication Date: {pubdate}\n"
                    # paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)

                if not paper_sums:
                    return "No papers found matching the query."
                time.sleep(2.0)  # Adhere to arXiv API politeness guidelines
                return "\n".join(paper_sums)

            except Exception as e:
                print(f"Error during arXiv search (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(
                        3 * retry_count
                    )  # Exponential backoff, as per arXiv guidelines (3s minimum)
                    continue
        return None

    def retrieve_full_paper_text(self, paper_id: str, max_len: int = 50000) -> str:
        """Retrieves and extracts the full text of a specified arXiv paper.

        The paper is identified by its arXiv ID. The method downloads the PDF,
        extracts text from all pages, and then deletes the local PDF file.
        The extracted text is truncated to a specified maximum length.

        Args:
            paper_id (str): The arXiv ID of the paper to retrieve (e.g., "2303.08774v1" or "2303.08774").
            max_len (int, optional): The maximum number of characters of the
                                     extracted text to return. Defaults to 50000.

        Returns:
            str: The extracted text content of the paper, truncated to `max_len`.
                 Returns "EXTRACTION FAILED" if PDF text extraction encounters an error.
                 Returns "PAPER NOT FOUND" if the paper_id is invalid or paper not found.
        """
        pdf_text = ""
        download_filename = f"{paper_id.replace('/', '_')}.pdf"  # Use a more specific filename
        try:
            # Use arxiv.Client() for fetching a single paper by ID for clarity
            client = arxiv.Client()
            search_by_id = arxiv.Search(id_list=[paper_id])
            paper_results = list(client.results(search_by_id))

            if not paper_results:
                return "PAPER NOT FOUND"
            paper = paper_results[0]

            # Download the PDF to the PWD with a custom filename.
            paper.download_pdf(filename=download_filename)

            # Creating a pdf reader object
            reader = PdfReader(download_filename)
            # Iterate over all the pages
            for page_number, page in enumerate(reader.pages, start=1):
                # Extract text from the page
                try:
                    text = page.extract_text()
                    if text:  # Ensure text is not None
                        pdf_text += f"--- Page {page_number} ---\n"
                        pdf_text += text
                        pdf_text += "\n\n"
                except Exception as extraction_error:
                    print(
                        f"Error extracting text from page {page_number} of {download_filename}: {extraction_error}"
                    )
                    # Continue to next page if one page fails
                    continue

            if not pdf_text:  # If no text could be extracted from any page
                raise Exception("No text extracted from PDF.")

        except arxiv.arxiv.InvalidId as e:
            print(f"Invalid arXiv ID: {paper_id}. Error: {e}")
            return "PAPER NOT FOUND (Invalid ID)"
        except Exception as e:
            print(f"Error retrieving or processing paper {paper_id}: {e}")
            if os.path.exists(download_filename):
                os.remove(download_filename)
            time.sleep(1.0)  # Short pause after file operation
            return "EXTRACTION FAILED"
        finally:
            if os.path.exists(download_filename):
                try:
                    os.remove(download_filename)
                except OSError as e:
                    print(f"Error deleting temporary file {download_filename}: {e}")
            time.sleep(1.0)  # Short pause, good practice after operations

        return pdf_text[:max_len]
