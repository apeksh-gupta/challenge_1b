import fitz 
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import statistics

class PDFProcessor:
    def __init__(self):
        self.heading_patterns = [
            r'^(Chapter|CHAPTER)\s+\d+',
            r'^\d+\.\s+',  
            r'^\d+\.\d+\s+',  
            r'^\d+\.\d+\.\d+\s+', 
            r'^[A-Z][A-Z\s]{2,}$', 
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', 
        ]
        
    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured outline from PDF"""
        doc = fitz.open(pdf_path)
        
        try:

            text_blocks = self._extract_text_blocks(doc)
            
            title = self._detect_title(text_blocks)

            headings = self._detect_headings(text_blocks)

            outline = self._build_outline(headings)
            
            return {
                "title": title,
                "outline": outline
            }
            
        finally:
            doc.close()
    
    def _extract_text_blocks(self, doc) -> List[Dict]:
        """Extract text blocks with formatting metadata"""
        blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]

            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if "lines" in block: 
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                blocks.append({
                                    "text": text,
                                    "page": page_num + 1,
                                    "font_size": span["size"],
                                    "font_name": span["font"],
                                    "flags": span["flags"], 
                                    "bbox": span["bbox"],
                                    "color": span.get("color", 0)
                                })
        
        return blocks
    
    def _detect_title(self, blocks: List[Dict]) -> str:
        """Detect document title"""
        if not blocks:
            return "Untitled Document"

        first_page_blocks = [b for b in blocks if b["page"] <= 2]
        
        if not first_page_blocks:
            return "Untitled Document"

        max_font_size = max(b["font_size"] for b in first_page_blocks)

        title_candidates = [
            b["text"] for b in first_page_blocks 
            if b["font_size"] == max_font_size and len(b["text"]) > 3
        ]
        
        if title_candidates:
            title = max(title_candidates, key=len)
            return self._clean_text(title)

        for block in first_page_blocks:
            if len(block["text"]) > 10:
                return self._clean_text(block["text"])
        
        return "Untitled Document"
    
    def _detect_headings(self, blocks: List[Dict]) -> List[Dict]:
        """Detect headings using multiple signals"""
        if not blocks:
            return []

        font_sizes = [b["font_size"] for b in blocks]
        avg_font_size = statistics.mean(font_sizes)
        font_size_threshold = avg_font_size * 1.1
        
        headings = []
        
        for block in blocks:
            heading_level = self._classify_heading(block, avg_font_size)
            
            if heading_level:
                headings.append({
                    "level": heading_level,
                    "text": self._clean_text(block["text"]),
                    "page": block["page"],
                    "font_size": block["font_size"],
                    "confidence": self._calculate_confidence(block, avg_font_size)
                })

        headings.sort(key=lambda x: (-x["confidence"], x["page"]))
        return self._deduplicate_headings(headings)
    
    def _classify_heading(self, block: Dict, avg_font_size: float) -> str:
        """Classify text block as heading level"""
        text = block["text"]
        font_size = block["font_size"]
        flags = block["flags"]

        if len(text) < 3 or len(text) > 200:
            return None

        if self._looks_like_body_text(text):
            return None

        size_ratio = font_size / avg_font_size

        is_bold = flags & 2**4 

        pattern_score = self._match_heading_patterns(text)

        if size_ratio > 1.5 or pattern_score > 0.8:
            return "H1"
        elif size_ratio > 1.2 or (is_bold and pattern_score > 0.5):
            return "H2"
        elif size_ratio > 1.1 or (is_bold and pattern_score > 0.3):
            return "H3"
        elif is_bold and self._looks_like_heading(text):
            return "H3"
        
        return None
    
    def _match_heading_patterns(self, text: str) -> float:
        """Match text against heading patterns"""
        score = 0.0
        
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                score += 0.3

        if text.isupper() and len(text) < 50:
            score += 0.2
        
        if re.match(r'^[A-Z]', text) and not text.endswith('.'):
            score += 0.1
        
        if len(text.split()) <= 8: 
            score += 0.1
        
        return min(score, 1.0)
    
    def _looks_like_heading(self, text: str) -> bool:

        words = text.split()
        if len(words) <= 6 and all(word[0].isupper() for word in words if word):
            return True

        if text[0].isupper() and not text.endswith(('.', '!', '?')):
            return True
        
        return False
    
    def _looks_like_body_text(self, text: str) -> bool:

        if len(text) > 100 and '.' in text:
            return True
        
        if text.count('.') > 1:
            return True

        body_indicators = ['the', 'and', 'that', 'this', 'with', 'from']
        words = text.lower().split()
        if len(words) > 5 and any(word in body_indicators for word in words[:3]):
            return True
        
        return False
    
    def _calculate_confidence(self, block: Dict, avg_font_size: float) -> float:
        """Calculate confidence score for heading classification"""
        confidence = 0.0
        
        size_ratio = block["font_size"] / avg_font_size
        confidence += min(size_ratio - 1.0, 0.5) * 2  

        if block["flags"] & 2**4:
            confidence += 0.3

        confidence += self._match_heading_patterns(block["text"]) * 0.5

        page_factor = max(0, 1.0 - (block["page"] - 1) * 0.1)
        confidence += page_factor * 0.2
        
        return min(confidence, 1.0)
    
    def _deduplicate_headings(self, headings: List[Dict]) -> List[Dict]:
        """Remove duplicate headings"""
        seen = set()
        unique_headings = []
        
        for heading in headings:

            key = (heading["text"].lower(), heading["page"])
            
            if key not in seen:
                seen.add(key)
                unique_headings.append(heading)
        
        return unique_headings
    
    def _build_outline(self, headings: List[Dict]) -> List[Dict]:
        """Build final outline structure"""
        outline = []
        
        for heading in headings:
            outline.append({
                "level": heading["level"],
                "text": heading["text"],
                "page": heading["page"]
            })
        
        outline.sort(key=lambda x: x["page"])
        
        return outline
    
    def _clean_text(self, text: str) -> str:

        text = re.sub(r'\s+', ' ', text).strip()

        text = re.sub(r'^[•\-\*]\s*', '', text)  
        
        return text
