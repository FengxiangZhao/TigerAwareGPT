import gradio as gr
import pandas as pd
import tenacity
import openai
import concurrent.futures

from tqdm import tqdm

from utils import *

SPLIT_THRESHOLD = 3072
API_URL = "https://api.openai.com/v1/chat/completions"
my_api_key = "sk-xxxxxxxxxxxxxxxxxxxxx"
INITIAL_SURVEY_CODING_SYSTEM_PROMPT = "You are a researcher trying to coding the responses from given survey responses"
FINAL_SURVEY_CODING_SYSTEM_PROMPT = "You are a researcher trying to summarize the results you have coded from the survey responses."
RESPONSE_CODING_SYSTEM_PROMPT = "You are researcher who want to identified if specific survey response is related to given key words"

ASSISTANT_PROMPT = "Below are the survey responses. Please code the responses accordingly. Each line is a response."
DEBUG = False

customCSS = """
code {
    display: inline;
    white-space: break-spaces;
    border-radius: 6px;
    margin: 0 2px 0 2px;
    padding: .2em .4em .1em .4em;
    background-color: rgba(175,184,193,0.2);
}
pre code {
    display: block;
    white-space: pre;
    background-color: hsla(0, 0%, 0%, 72%);
    border: solid 5px var(--color-border-primary) !important;
    border-radius: 10px;
    padding: 0 1.2rem 1.2rem;
    margin-top: 1em !important;
    color: #FFF;
    box-shadow: inset 0px 8px 16px hsla(0, 0%, 0%, .2)
}
"""


class SurveyCoder:
    def __init__(self, file):
        # Initialize the dataframe
        self.split_df = None  # A list of dataframes that separated by the threshold
        self.summary_list = None  # A list of summaries
        self.final_summary = ""  # The final summary
        self.df = pd.DataFrame(columns=['content', 'num_token'])
        self.parse_txt(file)
        self.token_usage = 0
        self.coding_token_usage = 0
        self.coding_df = None


    def parse_txt(self, file):
        # Parse the txt file and store the content in a dataframe
        with open(file.name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                new_row = {'content': line, 'num_token': count_token(line)}
                new_df = pd.DataFrame(new_row, index=[0])
                self.df = pd.concat([self.df, new_df], ignore_index=True)
        shuffled_df = self.df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe
        self.split_df = split_dataframe(shuffled_df, SPLIT_THRESHOLD)

    def get_coding_df(self, key_words):
        self.coding_df = pd.DataFrame(columns=key_words)

        def process_row(index, row):
            print(f"coding {index}/{len(self.df)}")
            response = row[1]  # row[1] is the content
            response_coding = self.chat_response_coding(response, key_words)
            response_coding = parse_response_coding_output(response_coding, key_words)
            response_coding['content'] = response
            # new_df = pd.DataFrame(response_coding, index=[i])
            # self.coding_df = pd.concat([self.coding_df, new_df], ignore_index=True)
            return pd.DataFrame(response_coding, index=[index])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            rows = list(tqdm(executor.map(process_row, self.df.index, self.df.itertuples()), total=len(self.df)))
        self.coding_df = pd.concat(rows, ignore_index=True)
        return self.coding_df

    def summary_with_chat(self, number_of_bp):
        # Summarize the dataframe with API
        summary_list = []

        def process_df(df):
            text = dataframe_to_string(df)
            response_text = self.chat_conclusion(text, number_of_bp)
            key_words = extract_strings_in_square_brackets(response_text)
            return key_words

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            result_futures = [executor.submit(process_df, df) for df in self.split_df]

        for future in concurrent.futures.as_completed(result_futures):
            summary_list.extend(future.result())

        final_summary = self.chat_final_conclusion('\n'.join(summary_list), number_of_bp)
        final_summary = '\n'.join(extract_strings_in_square_brackets(final_summary))
        self.final_summary = final_summary
        return final_summary

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=10, max=30),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_conclusion(self, text, number_of_bp):
        openai.api_key = my_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                construct_system(INITIAL_SURVEY_CODING_SYSTEM_PROMPT),
                construct_initial_user(text, number_of_bp)
            ],
            temperature=1,
            top_p=1,
            n=1
        )
        result = ''
        self.token_usage += response.usage['total_tokens']
        print(f"{response.usage['total_tokens'] = }")
        for choice in response.choices:
            result += choice.message.content
        print("conclusion_result:\n", result)
        return result

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=10, max=30),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_final_conclusion(self, text, number_of_bp):
        openai.api_key = my_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                construct_system(FINAL_SURVEY_CODING_SYSTEM_PROMPT),
                construct_final_user(text, number_of_bp)
            ],
            temperature=1,
            top_p=1,
            n=1
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print("final_conclusion_result:\n", result)
        return result

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=10, max=30),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_response_coding(self, survey_response, key_words):
        openai.api_key = my_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                construct_system(RESPONSE_CODING_SYSTEM_PROMPT),
                # construct_response_coding_assistant(key_words),
                construct_survey_coding_user(survey_response, key_words)
            ],
            temperature=1,
            top_p=1,
            n=1
        )
        result = ''
        self.coding_token_usage += response.usage['total_tokens']
        for choice in response.choices:
            result += choice.message.content
        print(f"\n{survey_response = }\n", "response_coding_result:\n", result)
        return result


def upload_txt(uploaded_txt, number_of_bp):
    if not file:
        return "Please update the txt file!", "Please update the txt file!", None
    if not uploaded_txt.name.endswith('.txt'):
        return "Please upload a txt file!", "Please update the txt file!", None
    coder = SurveyCoder(uploaded_txt)
    est_token_usage = coder.df['num_token'].sum() + coder.df.shape[0] * 200  # 105 is the token usage for the initial prompt
    if not DEBUG:
        print(f'{coder.token_usage = }')
        return coder.summary_with_chat(number_of_bp), token_usage_check(coder.token_usage, est_token_usage), coder
    else:
        output_str = """
        [Addressing Traffic and Transportation];
        [Affordable Housing];
        [City Management and Communication];
        [Homelessness];
        [Taxes and Regulations];
        [City Planning and Growth];
        [Utility and Energy Issues];
        [Public Safety];
        [Cleanliness and Environment];
        [Environmental issues such as pollution and creek maintenance];
              """
        return '\n'.join(extract_strings_in_square_brackets(output_str)), token_usage_check(coder.token_usage, est_token_usage), coder

def reset_text_box(coder: SurveyCoder):
    if coder:
        return coder.final_summary
    else:
        return "Please upload the file first!"


def process_coding(user_input, coder: SurveyCoder, progress=gr.Progress(track_tqdm=True)):
    # coder.progress_bar = progress
    key_words = user_input.strip().splitlines()
    est_token_usage = coder.df['num_token'].sum() + coder.df.shape[0] * 200  # 200 is the token usage for the initial prompt
    print(f"process_coding...{key_words = }")
    coder.coding_df = coder.get_coding_df(key_words)
    return gr.DataFrame.update(value=coder.coding_df, visible=True), token_usage_check(coder.token_usage, est_token_usage, coder.coding_token_usage)


def export_csv(coding_df):
    coding_df.to_csv('coding.csv', index=False)
    return gr.File.update(value='coding.csv', visible=True, interactive=True)


title = "TigerAware Survey Coder"
with gr.Blocks(css=customCSS) as demo:
    gr.HTML(f"""<h1 align="center">{title} {'DEBUG' if DEBUG else ''}</h1>""")
    gr.HTML("<h2 align='center'>A survey coder that auto code the surveys</h2>")
    survey_coder = gr.State(None)
    with gr.Row():
        with gr.Column(scale=4):
            file = gr.File(label="upload the txt file")
        with gr.Column(scale=1):
            token_usage = gr.HTML(value=token_usage_check(), label="token usage")
    with gr.Row():
        with gr.Column(scale=4):
            number_of_points = gr.Slider(minimum=3, maximum=15, value=10, step=1,
                                         interactive=True, label="Max number of bullet points you want to output", )
        with gr.Column(scale=1):
            txt_upload_button = gr.Button(value="Upload", label="Upload", variant="primary")
    with gr.Row():
        keywords_output = gr.Textbox(lines=2, label="Key words", interactive=True)
    with gr.Row():
        reset_button = gr.Button(value="Reset", label="reset")
        coding_process_button = gr.Button(value="Process", label="theme_process_button", variant="primary")
    gr.HTML("<h2 align='center'>Quantify responses on basis of the coding</h2>")
    gr.HTML(
        "<div align='center'>Each response is rated with a range of 1~5, where 1 means unrelated and 5 means very related </div>")
    gr.HTML(
        "<div align='center'>It take some time to process the response. Around 100 responses per minute.</div>")
    with gr.Row():
        with gr.Column():
            coding_data = gr.DataFrame(label="data", interactive=False, visible=True)
    with gr.Row():
        with gr.Column(scale=1):
            export_button = gr.Button(value="Export CSV", label="Export", variant="primary")
        with gr.Column(scale=3):
            export_file = gr.File(label="csv", visible=False)

    txt_upload_button.click(upload_txt, inputs=[file, number_of_points], outputs=[keywords_output, token_usage, survey_coder])
    reset_button.click(reset_text_box, inputs=[survey_coder], outputs=keywords_output)
    coding_process_button.click(process_coding, inputs=[keywords_output, survey_coder], outputs=[coding_data, token_usage])
    export_button.click(export_csv, inputs=[coding_data], outputs=export_file)

# interface = gradio.Interface(fn=upload_pdf, inputs=ip, outputs="html", title=title, description=description)
# interface = gradio.Interface(upload_txt, inputs="file", outputs="html", title=title, description=description)
demo.title = title
# run the app
demo.queue(concurrency_count=8).launch(share=True)
