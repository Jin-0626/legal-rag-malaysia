"""
Web scraper module for Malaysian legal documents using BeautifulSoup4
"""
import logging
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LawScraper:
    """Scraper for Malaysian legal documents from LOM and Kehakiman portals"""
    
    def __init__(self, download_dir: str = None):
        """
        Initialize the web scraper
        
        Args:
            download_dir: Directory to save downloaded PDFs
        """
        self.download_dir = Path(download_dir) if download_dir else Path.cwd() / "downloads"
        self.download_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info(f"Scraper initialized. Download directory: {self.download_dir}")
    
    def scrape_lom_acts(self, url: str = "https://lom.agc.gov.my/", max_documents: int = 10) -> List[Dict]:
        """
        Scrape Acts from LOM (Laws of Malaysia) portal
        
        Args:
            url: LOM portal URL
            max_documents: Maximum number of documents to scrape
        
        Returns:
            List of dictionaries with document metadata
        """
        documents = []
        try:
            logger.info(f"Fetching LOM portal: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all PDF links
            links = soup.find_all('a', href=lambda href: href and '.pdf' in href.lower())
            logger.info(f"Found {len(links)} PDF links")
            
            # Extract document information
            for i, link in enumerate(links[:max_documents]):
                try:
                    href = link.get('href')
                    text = link.get_text(strip=True)
                    
                    if href:
                        full_url = urljoin(url, href)
                        filename = Path(href).name or f"document_{i+1}.pdf"
                        
                        documents.append({
                            "title": text or f"Document {i+1}",
                            "url": full_url,
                            "filename": filename,
                            "source": "LOM Portal"
                        })
                        logger.info(f"  {i+1}. {text[:50]}...")
                
                except Exception as e:
                    logger.warning(f"Error extracting link {i}: {e}")
            
            return documents
        
        except Exception as e:
            logger.error(f"Error scraping LOM portal: {e}")
            return documents
    
    def scrape_judgments(self, 
                        url: str = "https://www.kehakiman.gov.my/",
                        max_documents: int = 10) -> List[Dict]:
        """
        Scrape judgments from Kehakiman portal
        
        Args:
            url: Kehakiman portal URL
            max_documents: Maximum number of documents to scrape
        
        Returns:
            List of dictionaries with judgment metadata
        """
        documents = []
        try:
            logger.info(f"Fetching Kehakiman portal: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find judgment links (adjust selectors based on actual page structure)
            links = soup.find_all('a', href=lambda href: href and ('judgment' in href.lower() or 'case' in href.lower()))
            logger.info(f"Found {len(links)} judgment links")
            
            for i, link in enumerate(links[:max_documents]):
                try:
                    href = link.get('href')
                    text = link.get_text(strip=True)
                    
                    if href:
                        full_url = urljoin(url, href)
                        documents.append({
                            "title": text or f"Judgment {i+1}",
                            "url": full_url,
                            "source": "Kehakiman Portal"
                        })
                        logger.info(f"  {i+1}. {text[:50]}...")
                
                except Exception as e:
                    logger.warning(f"Error extracting judgment {i}: {e}")
            
            return documents
        
        except Exception as e:
            logger.error(f"Error scraping Kehakiman portal: {e}")
            return documents
    
    def download_pdf(self, url: str, filename: str = None) -> bool:
        """
        Download a PDF from URL
        
        Args:
            url: URL of the PDF
            filename: Optional filename to save as
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading PDF from {url}")
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            if not filename:
                filename = Path(url).name or "document.pdf"
            
            filepath = self.download_dir / filename
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress = int(100 * downloaded / total_size)
                            logger.debug(f"Progress: {progress}%")
            
            logger.info(f"OK: Saved to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"FAIL: Download failed: {e}")
            return False
    
    def download_multiple_pdfs(self, documents: List[Dict]) -> List[str]:
        """
        Download multiple PDFs
        
        Args:
            documents: List of document dictionaries with 'url' and optional 'filename'
        
        Returns:
            List of successfully downloaded file paths
        """
        downloaded = []
        
        for doc in documents:
            url = doc.get('url')
            filename = doc.get('filename')
            
            if url:
                if self.download_pdf(url, filename):
                    downloaded.append(str(self.download_dir / (filename or Path(url).name)))
        
        return downloaded
