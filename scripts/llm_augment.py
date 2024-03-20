'''
This script runs a data augmentation script by querying a Mixtral LLM model available on the internet

Steps to run:

1. Download dataset from https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset in case its not already done
2. Save it in a directory called dataset (but don't commit the dataset)
3. Run the script using your runid

python llm_augment.py --runid {1-6}
'''

import json
import requests


def ping_llm():
    prompt = """
    Context: I am trying to perform data augmentation on 4-way text classification of the LUN Dataset which has the following class labels: 1-Satire, 2-Hoax, 3-Propaganda, 4-Reliable News.
    Description: I will give you example sentences where each sentence is enclosed by a special token {SOS} along with the original class label enclosed in {CL}.
    
    For each example sentence do the following:
    1. Paraphrased text with loosing any meaning while still being in the style as the class label mentioned. Lets call it "paraphrased".
    2. Also produce a justification (using your knowledge of grammar and human psychology) for why the original text was in that style/class label. Call it "original_justification"
    3. Do the same justification for why the paraphrased text you generated in #1 is in the same style as well. Call it "paraphrased_justification".
    
    Produce your answer for each example in this format exactly:
    {example_number}|{paraphrased}|{original_justification}|{paraphrased_justification}<EOA>
    
    I plan to use "|" and "<EOA>" tokens you generate to help me parse your answer.
    
    Paraphrase the following:
    """
    try:

        example = "{SOS}The writers of the HBO series The Sopranos took another daring storytelling step by killing off 10 million fans during the seventh season's premiere episode Sunday night. 'This was definitely a bold choice, one that producers of the show would have never thought of making five years ago,' said New York Times television critic Virginia Heffernan, who noted that the move was hinted at in a season-five episode in which Tony dreamt he was riding a horse through his house. 'But now that I look back, this was strongly foreshadowed throughout all of last season.' Industry insiders predicted that the show's producers would try to bring at least some fans back for the series finale, which may come as early as May. {SOS}|{CL}Satire{CL}"

        url = "https://iiced-mixtral-46-7b-fastapi.hf.space/generate/"
        data = {
            "prompt": prompt+example,
            "history": [],
            "system_prompt": "You are expected to help in augmenting a text based dataset for 4-way text classification"
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, data=json.dumps(data), headers=headers)
        return response.json()
    except Exception as e:
        print(f"Failed with exception {e}")
        return


if __name__ == '__main__':
    response = ping_llm()
    print(response.json())
