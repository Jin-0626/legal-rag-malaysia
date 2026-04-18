"""
Document processing module for converting PDFs to markdown with section awareness
"""
import os
import re
from pathlib import Path
from typing import List, Tuple
import json

class LegalDocumentProcessor:
    """Process Malaysian legal documents with section awareness"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_pattern = re.compile(r'(Section\s+\d+\w*|Part\s+[\dIVX]+)')
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            return text
        except ImportError:
            print("PyMuPDF not installed. Install with: pip install pymupdf")
            return ""
    
    def split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text by legal sections with metadata"""
        sections = []
        current_section = None
        current_content = ""
        
        lines = text.split('\n')
        for line in lines:
            if self.section_pattern.match(line.strip()):
                if current_section and current_content.strip():
                    sections.append((current_section, current_content.strip()))
                current_section = line.strip()
                current_content = ""
            else:
                current_content += line + "\n"
        
        if current_section and current_content.strip():
            sections.append((current_section, current_content.strip()))
        
        return sections
    
    def chunk_with_overlap(self, text: str) -> List[str]:
        """Create chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, pdf_path: str, act_number: str) -> List[dict]:
        """Process a single legal document"""
        print(f"Processing: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Failed to extract text from {pdf_path}")
            return []
        
        # Split by sections
        sections = self.split_by_sections(text)
        
        # Create chunks with metadata
        chunks = []
        for section_name, section_text in sections:
            section_chunks = self.chunk_with_overlap(section_text)
            for i, chunk in enumerate(section_chunks):
                chunks.append({
                    "act": act_number,
                    "section": section_name,
                    "chunk_id": f"{act_number}_{section_name.replace(' ', '_')}_{i}",
                    "content": chunk,
                    "source": pdf_path
                })
        
        print(f"Created {len(chunks)} chunks from {pdf_path}")
        return chunks
    
    def save_processed_chunks(self, chunks: List[dict], output_path: str):
        """Save processed chunks to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {output_path}")
