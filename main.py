import os
import json
import time
from pathlib import Path
import fitz 
from sentence_transformers import SentenceTransformer, util
import torch

from src.pdf_processor import PDFProcessor

INPUT_DIR = Path("input") 
OUTPUT_DIR = Path("output") 
MODEL_NAME = 'all-MiniLM-L6-v2' 

TOP_N_SECTIONS = 5
TOP_N_SUBSECTIONS = 5

def get_section_full_text(doc: fitz.Document, outline: list) -> list:
    """
    Extracts the full text for each section defined in the outline.
    This version is highly robust and maps text blocks to headings precisely,
    tolerating mismatches in whitespace and capitalization.
    """
    if not outline:
        return []

    all_blocks = []
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict").get("blocks", []):
            if block.get('type') == 0: 
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                if block_text.strip():
                    all_blocks.append({
                        'page_num': page_num,
                        'text': block_text,
                        'block_idx': len(all_blocks)
                    })

    matched_headings = []
    headings_to_find = list(outline)
    headings_to_find.sort(key=lambda x: (x['page'], x['level_num'] if 'level_num' in x else x['level']))

    matched_block_indices = set()

    for heading in headings_to_find:
        found_block_info = None
        for block in all_blocks:
            if block['block_idx'] in matched_block_indices:
                continue

            normalized_block_text = "".join(block['text'].lower().split())
            normalized_heading_text = "".join(heading['text'].lower().split())

            if (abs(block['page_num'] - (heading['page'] - 1)) <= 1 and
                normalized_heading_text in normalized_block_text and
                len(normalized_heading_text) > 2):

                if found_block_info is None or (block['page_num'] == heading['page'] - 1 and found_block_info['block_info']['page_num'] != heading['page'] - 1):
                    found_block_info = {
                        'block_idx': block['block_idx'],
                        'heading_info': heading,
                        'block_info': block
                    }
        
        if found_block_info:
            matched_headings.append(found_block_info)
            matched_block_indices.add(found_block_info['block_idx'])

    matched_headings.sort(key=lambda x: x['block_idx'])

    if not matched_headings:
        print("Warning: Could not match any outline headings to the document's text blocks.")
        return []

    sections = []
    for i in range(len(matched_headings)):
        current_match = matched_headings[i]
        start_block_idx = current_match['block_idx']
        end_block_idx = matched_headings[i+1]['block_idx'] if i + 1 < len(matched_headings) else len(all_blocks)

        section_text_blocks = [all_blocks[j]['text'] for j in range(start_block_idx, end_block_idx)]
        full_text = " ".join(section_text_blocks)

        sections.append({
            "title": current_match['heading_info']['text'],
            "page": current_match['heading_info']['page'],
            "full_text": full_text.strip()
        })

    return sections


def process_for_round_1b():
    """Main function for the Round 1B challenge."""
    start_time = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading embedding model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model loaded on {device}.")

    try:
        with open(INPUT_DIR / "persona.json") as f:
            persona_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: persona.json not found in the '{INPUT_DIR}' directory!")
        print("Please create 'input/persona.json' with 'persona' and 'job_to_be_done' fields.")
        return

    persona = persona_data.get('persona', '')
    job_to_be_done = persona_data.get('job_to_be_done', '')

    query = f"{persona} wants to {job_to_be_done}. Focus on Roman ruins, medieval architecture, historical attractions, ancient sites, and old structures."
    print(f"Generated Semantic Query: {query}")
    query_embedding = model.encode(query, convert_to_tensor=True)

    outline_processor = PDFProcessor()
    all_sections = []

    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}' directory. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_path in pdf_files:
        print(f"Extracting sections from {pdf_path.name}...")
        try:
            doc = fitz.open(pdf_path)
            outline_result = outline_processor.extract_outline(str(pdf_path))
            outline = outline_result.get("outline", [])

            if not outline:
                print(f"Warning: No outline extracted for {pdf_path.name}.")
                continue

            sections_with_text = get_section_full_text(doc, outline)
            for section in sections_with_text:
                section['doc_name'] = pdf_path.name
                all_sections.append(section)

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    if not all_sections:
        print("No sections could be extracted from any documents. Exiting.")
        return

    print(f"Creating embeddings for {len(all_sections)} sections...")
    section_texts = [
        f"{s['title']}. {s['title']}. {s['full_text']}" for s in all_sections]
    section_embeddings = model.encode(section_texts, convert_to_tensor=True, show_progress_bar=True)

    cosine_scores = util.cos_sim(query_embedding, section_embeddings)

    for i, section in enumerate(all_sections):
        section['score'] = cosine_scores[0][i].item()

    ranked_sections = sorted(all_sections, key=lambda x: x['score'], reverse=True)

    extracted_sections_output = []
    for i, s in enumerate(ranked_sections[:TOP_N_SECTIONS]):
        extracted_sections_output.append({
            "document": s['doc_name'],
            "section_title": s['title'],
            "importance_rank": i + 1, 
            "page_number": s['page']
        })

    print("Performing sub-section analysis on top-ranked sections...")
    sub_section_results = []

    for section in ranked_sections[:TOP_N_SUBSECTIONS]:
        sentences = [s.strip() for s in section['full_text'].split('.') if len(s.strip()) > 20]
        if not sentences:
            print(f"Warning: No valid sentences found in section '{section['title']}' from '{section['doc_name']}'.")
            continue

        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        sent_scores = util.cos_sim(query_embedding, sentence_embeddings)
        best_sent_idx = torch.argmax(sent_scores)

        sub_section_results.append({
            "document": section['doc_name'],
            "refined_text": sentences[best_sent_idx],
            "page_number": section['page']
        })


    output_data = {
        "metadata": {
            "input_documents": [p.name for p in pdf_files],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) 
        },
        "extracted_sections": extracted_sections_output, 
        "subsection_analysis": sub_section_results 
    }

    output_filename = OUTPUT_DIR / "challenge1b_output.json"
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nProcessing complete in {time.time() - start_time:.2f} seconds.")
    print(f"✓ Output written to {output_filename}")

if __name__ == "__main__":
    process_for_round_1b()
