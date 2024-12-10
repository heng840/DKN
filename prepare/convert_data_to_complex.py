import json
import re
# Define the expanded templates
templates = {
    "plays for": "_X_, renowned for their career in sports, plays for a team or club. This association is a significant part of their professional journey, reflecting their skill and dedication in their field.",
    "holds the position of": "_X_, a key figure in their domain, holds an important position. This role is marked by responsibilities and influence, shaping the direction and success of their organization or country.",
    "attended": "_X_, known for their intellectual and professional achievements, attended a notable institution. Their time at this institution played a crucial role in shaping their career and perspectives.",
    "is the chair of": "As the chair of _X_, this individual is a prominent figure in the business world. This position highlights their leadership and strategic vision in guiding the company towards growth and innovation.",
    "is a member of the": "_X_, a figure of significant influence and ideals, is a member of a notable group or organization. Their membership reflects their commitment to certain values and goals, contributing to the group's impact.",
    "works for": "_X_ works for a notable company or organization, playing a pivotal role in driving innovation and excellence in their field.",
    "is the head of the government of": "As the head of the government of _X_, this political figure plays a central role in shaping the nation's policies and international relations.",
    "is the head coach of": "The head coach of _X_ is known for their expertise in coaching, with leadership and strategic skills crucial in steering the team to success.",
    "is owned by": "_X_, a significant player in its sector, is owned by a larger entity. This ownership structure is key to understanding the company's operations and market influence."
}

# The function to rewrite sentences
# def rewrite_sentence_corrected(sentence):
#     # Adjusted to match sentences like 'Valentino Rossi plays for _X_.'
#     match = re.search(r'(.+?) (plays for|holds the position of|attended|is the chair of|is a member of the|works for|is the head of the government of|is the head coach of|is owned by) _X_.', sentence)
#     if match:
#         entity, relation = match.groups()
#         return templates.get(relation, "Relation not defined").format(entity=entity)
#     else:
#         return "Could not match the sentence with the template."
def rewrite_sentence_corrected(sentence):
    # Regular expression to match both cases
    match = re.search(r'(_X_ )?(plays for|holds the position of|attended|is the chair of|is a member of the|works for|is the head of the government of|is the head coach of|is owned by) (.+?)( _X_)?\.', sentence)

    if match:
        prefix, relation, entity, suffix = match.groups()

        # Determine if _X_ is at the beginning or end
        if prefix:  # _X_ at the beginning
            return templates.get(relation, "Relation not defined").format(entity=entity)
        else:       # _X_ at the end
            return templates.get(relation, "Relation not defined").format(entity=entity)
    else:
        return "Could not match the sentence with the template."

# Function to process the JSONL file
def rewrite_jsonl_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            # Parse the JSON line
            json_data = json.loads(line)
            original_sentence = json_data.get('query', '')

            # Rewrite the sentence
            rewritten_sentence = rewrite_sentence_corrected(original_sentence)

            # Store both original and rewritten sentences
            output_json_data = {"original": original_sentence, "rewritten": rewritten_sentence}
            output_file.write(json.dumps(output_json_data) + '\n')
output_file_path = '/home/chenyuheng/KN2/kn2/Templama/train_template_rewritten.jsonl'
input_file_path = '/home/chenyuheng/KN2/kn2/Templama/train_convert_tmp.jsonl'
rewrite_jsonl_file(input_file_path, output_file_path)

def match_and_replace(input_train_file_path, input_rewrite_file_path, output_file_path):
    # Load rewritten sentences into a dictionary for easy lookup
    rewritten_sentences = {}
    with open(input_rewrite_file_path, 'r') as rewrite_file:
        for line in rewrite_file:
            data = json.loads(line)
            original = data['original']
            rewritten = data['rewritten']
            rewritten_sentences[original] = rewritten

    # Process the training file and replace queries
    with open(input_train_file_path, 'r') as train_file, open(output_file_path, 'w') as output_file:
        for line in train_file:
            data = json.loads(line)
            original_query = data['query']
            if original_query in rewritten_sentences:
                data['query'] = rewritten_sentences[original_query]  # Replace with rewritten query
            output_file.write(json.dumps(data) + '\n')

# Example usage
input_train_file_path = '/home/chenyuheng/KN2/kn2/Templama/train.jsonl'
input_rewrite_file_path = output_file_path
final_output_file_path = '/home/chenyuheng/KN2/kn2/Templama/updated_train.jsonl'
match_and_replace(input_train_file_path, input_rewrite_file_path, final_output_file_path)
