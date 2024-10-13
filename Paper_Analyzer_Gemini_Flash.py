import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
from streamlit.components.v1 import html

# --- Configuration ---
os.environ["API_KEY"] = "AIzaSyCOhsh-JWBd6B006GA0UgdIW6wRcNon7lk"  # Replace with your actual API key
genai.configure(api_key=os.environ["API_KEY"])
AGENTS_OPTIONS = ["Paper Analyzer"]

# Use Google Gemini model
# Initialize the chat model
model = genai.GenerativeModel("gemini-1.5-flash")

NB_TOKENS = 1000000

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def Paper_Categories(paper_type):
    prompt = f"""
    
        Classify the following paper based on its type. The available categories are:

        - Scientific Paper
        - Academic Paper (Non-Scientific)
        - Review Article
        - White Paper
        - Position Paper
        - Conference Paper
        - Editorial or Opinion Piece
        - Technical Paper
        - Policy Paper
        - Case Study
        - Thesis or Dissertation
        - Journalistic Article
        - Legal Brief or Opinion

        For classification, consider the following characteristics:

        - Is it a research paper presenting an experiment or study? → Choose 'Scientific Paper'
        - Is it an academic paper without a focus on scientific experiments? → Choose 'Academic Paper (Non-Scientific)'
        - Does the paper review existing literature or research on a topic? → Choose 'Review Article'
        - Is the paper a formal report proposing solutions or offering recommendations? → Choose 'White Paper'
        - Does the paper present and argue for a specific stance or viewpoint? → Choose 'Position Paper'
        - Is the paper presented at an academic conference and discusses new findings or developments? → Choose 'Conference Paper'
        - Does the article reflect the personal opinions or editorial stance of the author? → Choose 'Editorial or Opinion Piece'
        - Is the paper technical and focused on detailed documentation or explanations of a method or process? → Choose 'Technical Paper'
        - Does the paper propose, analyze, or discuss policy-related matters? → Choose 'Policy Paper'
        - Does the paper explore a specific real-world scenario or problem in detail? → Choose 'Case Study'
        - Is the paper a thesis or dissertation from a degree program? → Choose 'Thesis or Dissertation'
        - Is the document a journalistic article covering news or analysis? → Choose 'Journalistic Article'
        - Is the paper a legal document analyzing a case or giving legal opinions? → Choose 'Legal Brief or Opinion'

        Paper Type: {paper_type}

        Your task: Identify which one and only one of the above categories the paper belongs to. Do not output anything else than the category name.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during processing: {e}"

def classify_paper(llm_classification):
    # List of possible categories
    categories = [
        "scientific paper",
        "academic paper (non-scientific)",
        "review article",
        "white paper",
        "position paper",
        "conference paper",
        "editorial or opinion piece",
        "technical paper",
        "policy paper",
        "case study",
        "thesis or dissertation",
        "journalistic article",
        "legal brief or opinion"
    ]
    
    # Convert the LLM classification to lowercase
    classification_lower = llm_classification.lower()

    # Check if the classification contains any of the category names
    for category in categories:
        if category in classification_lower:
            return category
    
    # If no category matches, return 'academic paper (non-scientific)'
    return "academic paper (non-scientific)"

def get_prompt_by_paper(category):
    # Dictionary mapping categories to their corresponding prompts
    category_prompts = {
        "scientific paper": "Analyze the following scientific paper, focusing on its hypothesis, research methods, data analysis, and findings. Evaluate the significance of the results in relation to the existing body of knowledge in the field, and assess the robustness of the experimental design, sample size, and statistical methods used. Also, critique the clarity and precision of the paper's presentation of results, the validity of the conclusions drawn, and any potential limitations or areas for further research.",
        "academic paper (non-scientific)": "Evaluate the following academic paper, paying close attention to its central argument, theoretical framework, and structure. Analyze how effectively the author supports their thesis using evidence, logical reasoning, and references to previous literature. Critique the quality of the argumentation, the breadth of perspectives considered, and the rigor of the analysis. Additionally, assess the paper’s clarity, coherence, and scholarly contribution to the field.",
        "review article": "Analyze this review article, focusing on its synthesis of existing research and how well it summarizes the state of the field. Evaluate the scope and depth of the literature reviewed, the critical analysis of different studies, and the identification of gaps or inconsistencies in the research. Critique the article's ability to provide a comprehensive overview, its objectivity, and the author’s insight into future directions for research.",
        "white paper": "Critique the following white paper, evaluating its presentation of a problem and proposed solutions. Analyze the clarity of its objectives, the depth of research supporting its claims, and the persuasiveness of its recommendations. Assess the paper’s organization, the quality of the evidence provided, and the appropriateness of its proposed implementation strategies. Consider the potential impact of the white paper on policy, industry, or practice, and identify any possible biases or limitations.",
        "position paper": "Analyze the following position paper by examining the clarity of the stance presented, the strength of the supporting arguments, and the use of evidence. Evaluate the paper’s engagement with counterarguments, the soundness of its reasoning, and its alignment with the current state of knowledge in the field. Critique the persuasiveness of the author's position, the ethical considerations raised, and any potential implications or broader impacts of the viewpoint.",
        "conference paper": "Examine this conference paper, focusing on its presentation of novel research or developments within the field. Evaluate the originality and significance of the research question, the rigor of the methods used, and the clarity of the findings. Assess the relevance of the paper to the conference's theme, its contribution to ongoing academic discourse, and its potential for sparking further research or collaboration. Critique the organization and conciseness of the paper given the conference format.",
        "editorial or opinion piece": "Critique this editorial or opinion piece by evaluating the strength of its argument, the quality of the evidence or examples provided, and the tone and style of the writing. Assess the clarity of the author’s viewpoint, their engagement with opposing perspectives, and the persuasive impact of the piece on its intended audience. Analyze any potential biases, the soundness of the logic, and the overall effectiveness of the article in shaping public or expert opinion.",
        "technical paper": "Analyze the following technical paper with a focus on its detailed description of processes, technologies, or methods. Evaluate the technical accuracy of the information, the clarity of explanations, and the comprehensiveness of the documentation. Assess the practical applicability of the technology or method presented, as well as any limitations or challenges highlighted by the authors. Critique the paper’s structure, the logical flow of the technical information, and its relevance to practitioners or researchers in the field.",
        "policy paper": "Evaluate this policy paper by analyzing the clarity of its objectives, the thoroughness of the research supporting its recommendations, and the feasibility of the proposed policy changes. Assess the strength of the evidence provided, the paper’s engagement with alternative policy options, and the potential impact of the recommendations on stakeholders. Critique the paper’s structure, the logic behind its arguments, and its contribution to current policy debates or frameworks.",
        "case study": "Analyze the following case study by focusing on the context, problem identification, and methodology used to explore the issue. Evaluate the depth of the analysis, the relevance of the findings, and the applicability of the lessons learned to broader contexts. Assess how well the case study synthesizes data or experiences, the robustness of its conclusions, and its contribution to practical knowledge in the field. Critique the clarity of the narrative and any potential biases in the interpretation of the case.",
        "thesis or dissertation": "Critique the following thesis or dissertation by examining its research question, literature review, methodology, and analysis. Evaluate the originality and significance of the research, the rigor of the data collection and analysis, and the clarity of the argumentation. Assess the quality of the discussion and conclusions, the contribution to academic knowledge, and the overall structure and coherence of the work. Identify any gaps or limitations in the study and suggest areas for future research.",
        "journalistic article": "Analyze the following journalistic article, focusing on the clarity of its reporting, the accuracy of the facts presented, and the quality of the sources cited. Evaluate the objectivity of the article, the depth of the analysis, and its relevance to the current news cycle or public interest. Assess the effectiveness of the narrative, the balance between storytelling and fact-reporting, and any potential biases in the coverage. Critique the article’s impact on its intended audience and its contribution to public discourse.",
        "legal brief or opinion": "Evaluate this legal brief or opinion by analyzing the clarity of the legal argument, the use of precedents, and the strength of the reasoning. Assess the thoroughness of the legal research, the interpretation of statutes or case law, and the soundness of the conclusions drawn. Critique the organization of the argument, the persuasiveness of the legal reasoning, and the relevance of the brief or opinion to the specific legal issue at hand. Consider any broader implications of the legal interpretation or ruling."
    }
    
    # Return the corresponding prompt, or a default one if the category is not recognized
    return category_prompts.get(category, category_prompts["academic paper (non-scientific)"])


def copy_to_clipboard_button(text_to_copy):
    html(f"""
        <button id="copyButton">Copy to Clipboard</button>
        <script>
            const copyButton = document.getElementById('copyButton');
            const textToCopy = `{text_to_copy}`;  
            copyButton.addEventListener('click', () => {{
                navigator.clipboard.writeText(textToCopy).then(() => {{
                    copyButton.innerText = "Copied!";
                }}).catch(err => {{
                    copyButton.innerText = "Copy Failed!";
                }});
            }});
        </script>
    """)

# Function to calculate the number of tokens (approximation: 1 word ≈ 3 token)
def count_tokens(text):
    return len(text.split()) * 3

# Function to get the first X chunks that do not exceed NB_TOKENS
def get_chunks_within_token_limit(chunks, token_limit):
    selected_chunks = []
    total_tokens = 0
    last_chunk = ""

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)

        # Add the chunk if it doesn't push the total tokens over the limit
        if total_tokens + chunk_tokens <= token_limit:
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens
            last_chunk = chunk  # Track the last chunk added
        else:
            break

    return ' '.join(selected_chunks), last_chunk

def Paper_Analyzer(CoT_Prompt):
    # Limit the text to the first X chunks that do not exceed NB_TOKENS
    text_to_analyze, last_chunk = get_chunks_within_token_limit(chunks, NB_TOKENS)

    prompt = f"""
        {CoT_Prompt}
        
        Here's the content of this paper:
        
        {text_to_analyze}
    """
    try:
        response = model.generate_content(prompt)
        return response.text, last_chunk
    except Exception as e:
        return f"An error occurred during process: {e}"

# --- Streamlit App UI Enhancements ---
st.set_page_config(
    page_title="Analyze a PDF file",
    page_icon=":mag_right:",
    layout="wide",
)

# Hide Streamlit's default menu and footer for a cleaner look
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #007bff;'>Analyze Your PDF</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Your PDF Files", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="You can upload multiple PDFs at once"
)

if uploaded_files:
    with st.spinner("Processing..."):
        combined_text = ""
        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            combined_text += pdf_text + "\n"
        chunks = chunk_text(combined_text)
    st.success("Your PDFs have been processed successfully!")

    # Extract the first 2 chunks or fewer if not enough chunks
    num_chunks = min(2, len(chunks))
    first_chunks = ' '.join(chunks[:num_chunks])

# Paper analysis section
if st.button("Analyze PDF"):
    if uploaded_files:
        with st.spinner("Analyzing..."):
            # Use first_chunks instead of combined_text for classification
            CoT_Class = Paper_Categories(first_chunks)
            CoT_Standardized_Class = classify_paper(CoT_Class)
            CoT_Prompt = get_prompt_by_paper(CoT_Standardized_Class)

            # Display paper classification and analysis
            st.write(f"**Paper Category:** {CoT_Class}")
            st.write(f"**Standardized Category:** {CoT_Standardized_Class}")
            st.write(f"**Generated Prompt:** {CoT_Prompt}")

            # Run paper analysis on the selected chunks (limiting to NB_TOKENS)
            translation, last_chunk = Paper_Analyzer(CoT_Prompt + "In the different parts of your answer, bring results/numbers/quotes from the paper to back your claims. Be a throrough and thoughful expert." + " Finally, create an elevator pitch (3-4 sentences) that communicates the study's key points.")

            # Append the analysis result to session state messages
            st.session_state.messages.append({"role": "assistant", "content": translation})

            # Display the last chunk included in the token limit
            #st.subheader("Last chunk included in the token limit:")
            #st.write(last_chunk)
    else:
        st.warning("Please upload a PDF file before analyzing.")

# Clear history
if st.button("Clear History"):
    st.session_state.messages = []
    st.experimental_rerun()

# Display previous messages (if any)
if st.session_state.messages:
    for message in st.session_state.messages:
        st.write(f"{message['role'].capitalize()}: {message['content']}")

        # Copy to clipboard functionality for the assistant messages
        if message["role"] == "assistant":
            copy_to_clipboard_button(message["content"])
