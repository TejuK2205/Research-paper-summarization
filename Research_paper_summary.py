import streamlit as st
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline, BartTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Initialize BART model and tokenizer
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name, tokenizer=BartTokenizer.from_pretrained(model_name))

def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def split_text_into_chunks(text, max_tokens=64):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(word_tokenize(sentence))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_text(prompt, text, max_length=50, min_length=25, chunk_limit=5):
    if not text.strip():
        st.error("No text extracted from PDF.")
        return ""

    max_tokens = 64
    chunks = split_text_into_chunks(text, max_tokens)

    summaries = []
    for chunk in chunks[:chunk_limit]:
        try:
            summary_chunk = summarizer(prompt + chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary_chunk)
        except Exception as e:
            st.error(f"Error occurred during summarization: {e}")
            return ""

    summary = ' '.join(summaries)
    summary = '. '.join([sentence.capitalize() for sentence in summary.split('. ')])
    return summary.strip()

def calculate_rouge_scores(original, generated):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(original, generated)
        return scores
    except Exception as e:
        st.error(f"Error calculating ROUGE scores: {e}")
        return {}

def calculate_bleu_score(original, generated):
    try:
        original_tokens = word_tokenize(original)
        generated_tokens = word_tokenize(generated)
        score = sentence_bleu([original_tokens], generated_tokens)
        return score
    except Exception as e:
        st.error(f"Error calculating BLEU score: {e}")
        return 0.0

def human_evaluation(generated):
    try:
        readability = len(generated.split())
        coherence = len(sent_tokenize(generated))
        relevance = len(set(generated.split()))
        scores = {'readability': readability, 'coherence': coherence, 'relevance': relevance}
        return scores
    except Exception as e:
        st.error(f"Error evaluating human scores: {e}")
        return {'readability': 0, 'coherence': 0, 'relevance': 0}

def main():
    st.title("Prompt-based Research Paper Summarization and Comparison Tool")
    st.write("Upload a research paper (PDF) to generate and compare abstracts.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file_uploader_1")
    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)
        if text.strip():
            st.write("**Extracted Text:**")
            st.write(text[:5000] + "...")

            if 'generated_abstract' not in st.session_state:
                st.session_state['generated_abstract'] = ""

            prompt = st.text_area("Enter your prompt:", "Summarize the main findings of the research paper.", key="text_area_prompt")

            if st.button("Generate Abstract", key="generate_button"):
                with st.spinner("Generating abstract..."):
                    generated_abstract = summarize_text(prompt, text, max_length=50, min_length=25)
                    st.session_state['generated_abstract'] = generated_abstract

            if st.session_state['generated_abstract']:
                st.write("**Generated Abstract:**")
                st.write(st.session_state['generated_abstract'])

                original_abstract = st.text_area("Paste the original abstract here:", key="text_area_original_abstract")
                if original_abstract:
                    st.subheader("Evaluation Metrics:")
                    rouge_scores = calculate_rouge_scores(original_abstract, st.session_state['generated_abstract'])
                    st.write("**ROUGE Scores:**")
                    st.write(rouge_scores)

                    bleu_score = calculate_bleu_score(original_abstract, st.session_state['generated_abstract'])
                    st.write("**BLEU Score:**")
                    st.write(bleu_score)

                    human_scores = human_evaluation(st.session_state['generated_abstract'])
                    st.write("**Human Evaluation Scores:**")
                    st.write(human_scores)

                else:
                    st.warning("Please enter the original abstract for evaluation.")
            else:
                st.warning("Click 'Generate Abstract' after entering the prompt.")
        else:
            st.warning("No text extracted from PDF. Please check the file and try again.")

if __name__ == "__main__":
    main()
