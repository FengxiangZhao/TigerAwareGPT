import re
import tiktoken


def count_token(input_str):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    length = len(encoding.encode(input_str))
    return length


def split_dataframe(df, threshold):
    sub_dfs = []
    current_sum = 0
    start_index = 0

    for i in range(len(df)):
        current_sum += df.loc[i, 'num_token']

        if current_sum > threshold:
            sub_df = df.iloc[start_index:i, :]
            sub_dfs.append(sub_df)
            current_sum = df.loc[i, 'num_token']
            start_index = i

    if current_sum > 0:
        sub_df = df.iloc[start_index:, :]
        sub_dfs.append(sub_df)
    print(f"Split {len(df)} rows into {len(sub_dfs)} dataframes.")
    return sub_dfs


def construct_text(role, text):
    return {"role": role, "content": text}


"""
text: the survey responses
number_of_bp: the number of bullet points in the survey
"""


def construct_initial_user(text, number_of_bp):
    initial_prompt = "Below are the survey responses. Please code the responses accordingly. Each line is a response."
    output_prompt = f"""                 
                     Output in the following format and replace 'xxx' with corresponding content or keywords. No more than {number_of_bp} aspects. Each line must be enclosed with '[' and '] and roughly 5-15 words':

                     [xxx];                     
                     [xxx];
                     [xxx];      

                     """
    return construct_text("user", initial_prompt + text + output_prompt)


def construct_final_user(text, number_of_bp):
    initial_prompt = """
                    Below are the key words and aspect you have already coded and identified. Summarize these coding accordingly." Each line is an aspect you have already identified.
                    Make sure there is no duplicate aspect in your summarization.
                    
                     """
    output_prompt = f"""                 
                     Output in the following format and replace 'xxx' with corresponding aspect or keywords. No more than {number_of_bp} aspects. Each aspect must be enclosed with '[' and ']':

                     [xxx];                     
                     [xxx];
                     [xxx];       

                     """
    return construct_text("user", initial_prompt + text + output_prompt)


# construct the prompt for the user to code the survey responses
# response: the survey response
# keywords: the keywords that the user has already identified
def construct_survey_coding_user(response, keywords):
    keywords_nl = '\n'.join([f'{k}=[number]' for k in keywords])
    prompt = f"""
    Please rate the degree of this statement is related to following keywords.
    The statement is: "{response}"
    Replace [number] with the degree of relativity in range of 1-5. 1 is not related and 5 is most related. 
    The output should be EXACTLY in following format. No other words or characters are allowed:
    {keywords_nl}
    
    """
    return construct_text("user", prompt)


def construct_system(text):
    return construct_text("system", text)


def construct_response_coding_assistant(keywords):
    keywords_nl = '\n'.join([f'{k}=[number]' for k in keywords])
    prompt = f"""
    Replace [number] with the degree of relativity in range of 1-5. 1 is not related and 5 is most related. 
    Do not include your comments or other words. The output should be EXACTLY in following format.
    {keywords_nl}
    """
    return construct_text("assistant", prompt)


# extract dataframe content to string
def dataframe_to_string(df):
    return ''.join(df['content'])


def extract_strings_in_square_brackets(text):
    pattern = r"\[([^\]]+)\]"
    matches = re.findall(pattern, text)
    return matches


def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


# get the most similar string from a list of strings based on Jaccard similarity
def get_similar_string_jaccard(key, str_list, threshold=0.7):
    for s in str_list:
        similarity = jaccard_similarity(key, s)
        if similarity > threshold:
            return s
    return None


# parse result into a dictionary
def parse_response_coding_output(input_string, keyword_list):
    output_dict = {}
    lines = input_string.strip().split('\n')
    for line in lines:
        try:
            key, value = line.replace("[", "").replace("]", "").replace(".", "").split(
                '=')  # remove square brackets and split
            key = key.strip()
            key_in_list = get_similar_string_jaccard(key, keyword_list)
            if key:
                output_dict[key_in_list] = int(value.strip())
            else:
                print(f"\n Keyword {key} not in keyword list. Ignored.")
                output_dict['system_message'] = f"Keyword {key} not in keyword list. Ignored."
        except ValueError:
            print(f"\n Invalid output format: [{line=}]  [{input_string = }] IGNORED.")
            output_dict['system_message'] = f"Invalid output format: {line}. {input_string = }. IGNORED"
    return output_dict


def token_usage_check(token_usage=None, est_token_usage=None, response_coding_token_usage=None):
    result = ""
    if token_usage:
        result += f"txt summarization token usage: {token_usage} "
    if est_token_usage:
        result += f"<br> Estimate Token usage for processing: {est_token_usage} "
    if response_coding_token_usage:
        result += f"<br> response coding token usage: {response_coding_token_usage}"
    return result


def update_progress_bar(current, total, progress):
    return f"Progress: {progress}%"
