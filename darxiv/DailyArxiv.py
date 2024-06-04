import datetime
import json
import os
import xml.etree.ElementTree as ET
import requests
from DailyOcr import ocr_folder
from DailyVec import convert_embeddings
import sys
import argparse

RSS_URL = "https://rss.arxiv.org/rss/cs"


class DailyArxiv:
    def __init__(self, rss_url=RSS_URL, self_query_date=None, paper_dir="papers", ocr_dir="ocr", metadata_dir="metadata", index_dir="index"):
        self.rss_url = rss_url
        if self_query_date is None:
            self.query_date = datetime.datetime.today().strftime("%Y-%m-%d")
        else:
            self.query_date = self_query_date
        self.metadata_dir = metadata_dir
        self.paper_dir = paper_dir
        self.ocr_dir = ocr_dir
        self.index_dir = index_dir
        if not os.path.exists(self.paper_dir):
            os.makedirs(self.paper_dir)
        if not os.path.exists(self.ocr_dir):
            os.makedirs(self.ocr_dir)
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

    def parse_rss(self, rss_url):
        response = requests.get(rss_url)
        if response.status_code != 200:
            print("Failed to fetch the RSS feed")
            return []

        root = ET.fromstring(response.content)
        items = []

        for item in root.findall(".//item"):
            paper = {
                "title": item.find("title").text,
                "link": item.find("link").text,
                "description": item.find("description").text,
                "guid": item.find("guid").text,
                "categories": [category.text for category in item.findall("category")],
                "announce_type": item.find("{http://arxiv.org/schemas/atom}announce_type").text,
                "rights": item.find("{http://purl.org/dc/elements/1.1/}rights").text,
                "creator": item.find("{http://purl.org/dc/elements/1.1/}creator").text,
            }
            items.append(paper)

        return items

    def fetch_metadata(self):
        date = self.query_date
        json_filename = f"papers_{date}.json"
        json_filename = os.path.join(self.metadata_dir, json_filename)
        if os.path.exists(json_filename):
            print(f"Metadata for {date} already exists, skipping")
            # load the existing metadata
            with open(json_filename, "r") as f:
                papers = json.load(f)
            return papers
        papers = self.parse_rss(self.rss_url)
        if papers:
            with open(json_filename, "w") as f:
                json.dump(papers, f, indent=4)
        else:
            print("No papers found or an error occurred")
        return papers

    def download_papers(self, metadata, only_new=True):
        print("Downloading papers")
        date = self.query_date
        download_dir = os.path.join(self.paper_dir, f"papers_{date}")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        paper_count = 0
        for paper in metadata:
            # guid format: "guid": "oai:arXiv.org:2405.18938v2"
            # parse the number after 'v', get the version number
            version_number = int(paper["guid"].split("v")[-1])
            if only_new and version_number != 1:
                # skip if only_new is True and the version number is not 1
                continue
            paper_count += 1
            paper_id = paper["guid"].split("/")[-1].split(":")[-1].replace("v", "_").replace(".", "-")
            # print(f"Downloading {paper['title']}")
            pdf_url = paper["link"].replace("/abs/", "/pdf/") + ".pdf"
            # print(f"PDF URL: {pdf_url}")
            pdf_link = f"https://arxiv-downloader.chivier.workers.dev/?pdfUrl={pdf_url}"
            # if exists, skip
            # print(pdf_link)
            if os.path.exists(f"{download_dir}/{paper_id}.pdf"):
                # print(f"File {paper_id} already exists, skipping")
                continue
            response = requests.get(pdf_link)
            # print(response)
            if response.status_code == 200:
                with open(f"{download_dir}/{paper_id}.pdf", "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {paper_id}")

        print("Download complete")
        print(f"Downloaded {paper_count} papers")
    
    def ocr_papers(self):
        date = self.query_date
        download_dir = f"papers_{date}"
        download_dir = os.path.join(self.paper_dir, download_dir)
        ocr_dir = f"ocr_{date}"
        ocr_dir = os.path.join(self.ocr_dir, ocr_dir)
        ocr_folder(download_dir, ocr_dir)
    
    def index_papers(self):
        date = self.query_date
        ocr_dir = f"ocr_{date}"
        ocr_dir = os.path.join(self.ocr_dir, ocr_dir)
        index_dir = f"index_{date}"
        index_dir = os.path.join(self.index_dir, index_dir)
        convert_embeddings(ocr_dir, index_dir)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Daily Arxiv")
    parser.add_argument("--rss_url", default="https://rss.arxiv.org/rss/cs", help="URL of the Arxiv RSS feed")
    parser.add_argument("--self_query_date", help="Query date in the format YYYY-MM-DD")
    parser.add_argument("--paper_dir", default=".", help="Directory to store downloaded papers")
    parser.add_argument("--ocr_dir", default=".", help="Directory to store OCR results")
    parser.add_argument("--metadata_dir", default=".", help="Directory to store metadata")
    parser.add_argument("--index_dir", default=".", help="Directory to store index")

    args = parser.parse_args()
    daily_arxiv = DailyArxiv(
        rss_url=args.rss_url,
        self_query_date=args.self_query_date,
        paper_dir=args.paper_dir,
        ocr_dir=args.ocr_dir,
        metadata_dir=args.metadata_dir,
        index_dir=args.index_dir
    )
    
    metadata = daily_arxiv.fetch_metadata()
    daily_arxiv.download_papers(metadata)
    daily_arxiv.ocr_papers()
    print("ocr done")
    daily_arxiv.index_papers()
    print("index done")
    
if __name__ == "__main__":
    # daily_arxiv = DailyArxiv(self_query_date="2024-05-31", paper_dir="data/papers", ocr_dir="data/ocr", metadata_dir="data/metadata")
    # metadata = daily_arxiv.fetch_metadata()
    # daily_arxiv.download_papers(metadata)
    main()
