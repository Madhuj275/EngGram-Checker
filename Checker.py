'''
Submitted
Group-12
B3-42 Madhuj Agrawal
B3-50 Om Mooley
'''

import sys
import spacy
import language_tool_python
import streamlit as st

# Increase recursion limit
sys.setrecursionlimit(10000)

# Load NLP tools
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

# --- CFG RULES ---
grammar_rules = {
    "S": [
        ["NP", "VP"],
        ["AUX", "NP", "VP"],
        ["WH", "AUX", "NP", "VP"],
        ["NP", "AUX", "ADJP"],
        ["NP", "AUX", "VP"],
        ["NP", "VP", "."],
        ["NP", "AUX", "VP", "PP"],
        ["NP", "VP", "PP"],
        ["PRP", "AUX", "ADJP"],
        ["DT", "AUX", "NP"],
        ["DT", "AUX", "ADJP"],
        ["WH", "AUX", "NP", "?"]
    ],
    "NP": [["DT", "NN"], ["DT", "JJ", "NN"], ["NN"], ["DT", "NN", "PP"], ["NNP"], ["PRP"], ["DT", "NNP"], ["DT", "PRP"]],
    "VP": [["VB", "NP"], ["VB", "PP"], ["VB", "TO", "VB"], ["VB"], ["AUX", "VB", "NP"], ["AUX", "ADJP"], ["VB", "ADJP"]],
    "PP": [["IN", "NP"]],
    "WH": [["WP"], ["WRB"]],
    "ADJP": [["JJ"], ["RB", "JJ"], ["ADVP", "JJ"]],
    ".": [["."]],
    "?": [["?"]]
}

first_sets = {
    "S": {"DT", "NN", "PRP", "AUX", "WH"},
    "NP": {"DT", "NN", "NNP", "PRP"},
    "VP": {"VB", "AUX"},
    "PP": {"IN"},
    "WH": {"WP", "WRB"},
    "ADJP": {"JJ", "RB"},
    ".": {"."},
    "?": {"?"}
}

follow_sets = {
    "S": {"$"},
    "NP": {"VP", "AUX", "PP", "ADJP", "$"},
    "VP": {"PP", ".", "$"},
    "PP": {"."},
    "WH": {"AUX"},
    "ADJP": {"."}
}

parsing_table = {}
for non_terminal, productions in grammar_rules.items():
    for production in productions:
        first_symbol = production[0]
        if first_symbol in first_sets:
            for terminal in first_sets[first_symbol]:
                parsing_table[(non_terminal, terminal)] = production

# --- TOKEN CATEGORY MAPPING ---
def get_token_category(token):
    if token.text in [".", "?"]:
        return token.text
    if token.dep_ == "aux":
        return "AUX"
    pos_mapping = {
        "PRP": "PRP", "DT": "DT", "JJ": "JJ", "RB": "RB", "NN": "NN", "NNS": "NN", "NNP": "NNP",
        "VB": "VB", "VBD": "VB", "VBG": "VB", "VBN": "VB", "VBP": "VB", "VBZ": "VB", "MD": "AUX",
        "IN": "IN", "TO": "TO", "WP": "WP", "WRB": "WRB"
    }
    return pos_mapping.get(token.tag_, None)


def parse_symbol(symbol, tokens, index, memo):
    key = (symbol, index)
    if key in memo:
        return memo[key]
    if symbol not in grammar_rules:
        if index < len(tokens) and tokens[index] == symbol:
            memo[key] = index + 1
            return index + 1
        return None
    for production in grammar_rules[symbol]:
        new_index = index
        success = True
        for sym in production:
            res = parse_symbol(sym, tokens, new_index, memo)
            if res is None:
                success = False
                break
            new_index = res
        if success:
            memo[key] = new_index
            return new_index
    return None

# --- LL(1) PARSER ---
def parse_sentence(sentence):
    doc = nlp(sentence)
    tokens = [get_token_category(token) for token in doc]
    tokens = [t for t in tokens if t is not None] + ["$"]
    stack = ["S"]
    index = 0
    
    while stack:
        top = stack.pop()
        current_token = tokens[index]
        
        if top in first_sets or top in follow_sets:  # Non-terminal
            if (top, current_token) in parsing_table:
                stack.extend(reversed(parsing_table[(top, current_token)]))
            else:
                return False  # Parsing failed
        elif top == current_token:  # Terminal match
            index += 1
        else:
            return False  # Parsing failed
    
    return index == len(tokens)

# --- DEPENDENCY CHECK ---
def dependency_check(sentence):
    doc = nlp(sentence)
    has_subject = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)
    has_verb = any(token.pos_ in ("VERB", "AUX") for token in doc)
    missing_prepositions = []
    verbs_requiring_prep = {"go", "arrive", "return", "travel", "listen", "depend", "believe"}
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in verbs_requiring_prep:
            dobj_children = [child for child in token.children if child.dep_ == "dobj"]
            prep_children = [child for child in token.children if child.dep_ == "prep"]
            if dobj_children and not prep_children:
                missing_prepositions.append((token.text, dobj_children[0].text))
    return has_subject and has_verb and not missing_prepositions

# --- FINAL GRAMMAR CHECK FUNCTION ---
def check_grammar(sentence):
    ll1_valid = parse_sentence(sentence)
    dep_valid = dependency_check(sentence)
    lt_errors = tool.check(sentence)
    result = f"LL1 Validity: {'✅' if ll1_valid else '❌'}\n"
    result += f"Dependency Check: {'✅' if dep_valid else '❌'}\n"
    result += f"LanguageTool Errors: {len(lt_errors)}\n"
    if ll1_valid and dep_valid and not lt_errors:
        result += "✅ Sentence is grammatically correct!"
    else:
        result += "❌ Sentence has issues:\n"
        if not ll1_valid:
            result += "- Structure violates grammar rules (CFG mismatch)\n"
        if not dep_valid:
            result += "- Missing subject, verb, or required prepositions\n"
        for error in lt_errors:
            result += f"- {error.message}\n"
    return result

# --- STREAMLIT UI ---
st.title("Grammar Checker")
sentence = st.text_area("Enter a sentence:")
if st.button("Check Grammar"):
    if sentence:
        result = check_grammar(sentence)
        st.text_area("Result:", result, height=200)
    else:
        st.warning("Please enter a sentence.")
