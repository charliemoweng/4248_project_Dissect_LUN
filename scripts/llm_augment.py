'''
This script runs a data augmentation script by querying a Mixtral LLM model available on the internet

Steps to run:

1. Download dataset from https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset in case its not already done
2. Save it in a directory called dataset (but don't commit the dataset)
3. Open fulltrain.csv and add this in the first line
"Label", "Text"
4. Run the script using your runid
python llm_augment.py --runid {1-6}
5. This will create a csv files ../dataset/{your_run_id)_train_aug.csv


'''

import json
import requests
import pandas as pd
from tqdm import tqdm
import argparse
import threading


'''
Current Class counts are:

Satire(1)           14047
Hoax(2)             6942
Propaganda(3)       17870
Reliable News(4)    9995

Goal is to get Hoax(2) and Reliable News(4) to around 14000
'''

class_to_label_mapper = {
    1: "Satire", 2: "Hoax", 3: "Propaganda", 4: "Reliable News"
}

label_to_class_mapper = {
    "Satire": 1, "Hoax": 2, "Propaganda": 3, "Reliable News": 4
}


def split_dataset(dataset):
    '''
    Goal is to save 6 different datasets with equal split of Hoax and Reliable News subsets as that is the split we want to increase

    :param dataset:
    :return:
    '''
    num_splits = 6
    counts = dataset["Label"].value_counts()

    # split hoax 6 ways
    hoax = dataset[dataset["Label"] == label_to_class_mapper["Hoax"]]
    hoax_split_size = len(hoax) // num_splits
    hoax_splits = [hoax[i * hoax_split_size:(i + 1) * hoax_split_size] for i in range(num_splits)]

    reliable = dataset[dataset["Label"] == label_to_class_mapper["Reliable News"]]
    reliable_split_size = len(reliable) // num_splits
    reliable_splits = [reliable[i * reliable_split_size:(i + 1) * reliable_split_size] for i in range(num_splits)]

    # saving the splits
    for i in range(num_splits):
        # Concatenate A_i and B_i
        combined_df = pd.concat([hoax_splits[i], reliable_splits[i]])

        # Save the split in the datasets folder
        combined_df.to_csv(f'../dataset/{i + 1}_train.csv', index=False)

        print(f'Saved {i + 1}_train.csv')


def augment(runid, split_dataset, no_of_hoax_to_generate_per_person, no_of_reliable_to_generate_per_person,
            hoax_generations,
            reliable_generations):
    hoax_dataset = split_dataset[split_dataset["Label"] == label_to_class_mapper["Hoax"]]
    reliable_dataset = split_dataset[split_dataset["Label"] == label_to_class_mapper["Reliable News"]]

    print(f"Generating {no_of_hoax_to_generate_per_person} paraphrased hoax sentences..")
    hoax_bar = tqdm(total=no_of_hoax_to_generate_per_person)
    while len(hoax_generations) < no_of_hoax_to_generate_per_person:
        # randomly select row
        row = hoax_dataset.sample(n=1)

        label = class_to_label_mapper[row["Label"].iloc[0]]
        text = row["Text"].iloc[0]

        result = ping_llm(label, text)
        hoax_generations.append(result)

        # update progress bar
        hoax_bar.update(1)

    hoax_bar.close()

    print(f"Generating {no_of_reliable_to_generate_per_person} reliable paraphrased sentences..")
    reliable_bar = tqdm(total=no_of_reliable_to_generate_per_person)
    while len(reliable_generations) < no_of_reliable_to_generate_per_person:
        # randomly select row
        row = reliable_dataset.sample(n=1)

        label = row["Label"].iloc[0]
        text = row["Text"].iloc[0]

        result = ping_llm(label, text)
        reliable_generations.append(result)

        # update progress bar
        reliable_bar.update(1)

    reliable_bar.close()

    print(f"Creating the new augmented dataset")
    hoax_generations_dataset = pd.DataFrame({
        "Label": [label_to_class_mapper["Hoax"]] * len(hoax_generations),
        "Text": hoax_generations
    })
    reliable_generations_dataset = pd.DataFrame({
        "Label": [label_to_class_mapper["Reliable News"]] * len(reliable_generations),
        "Text": reliable_generations
    })
    combined_df = pd.concat([split_dataset, hoax_generations_dataset, reliable_generations_dataset])

    # Save the split in the datasets folder
    combined_df.to_csv(f'../dataset/{runid}_train_aug.csv', index=False)

    print(f'Saved {runid}_train_aug.csv')

def ping_llm(label, text):
    try:
        system_prompt = """
        I am trying to perform data augmentation on 4-way text classification of the LUN Dataset which has the following class labels: Satire, Hoax, Propaganda, Reliable News. 
        In the prompt, you will be given both the text along with the class label. I want you to paraphrase the original text without loosing any meaning in the same style as the original class label. 
        In the prompt; the class label will be provided first, followed by a separator token ':' and finally followed by the original text.
        In your response make sure to return only the paraphrased text while being in the same style
        """
        prompt = f"{label}:{text}"

        data = {
            "prompt": prompt,
            "history": [],
            "system_prompt": system_prompt
        }
        url = "https://iiced-mixtral-46-7b-fastapi.hf.space/generate/"
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, data=json.dumps(data), headers=headers)
        return response.json()
    except Exception as e:
        print(f"Failed with exception {e}")
        return

    return "Test"


if __name__ == '__main__':
    dataset = pd.read_csv("../dataset/fulltrain.csv")

    # run this in case you still want to split it (it will most likely be the same thing)
    # split_dataset(dataset)

    label_counts = dataset["Label"].value_counts()

    no_of_hoax_to_generate_per_person = (label_counts[label_to_class_mapper["Satire"]] - label_counts[
        label_to_class_mapper["Hoax"]]) // 6
    no_of_reliable_to_generate_per_person = (label_counts[label_to_class_mapper["Satire"]] - label_counts[
        label_to_class_mapper["Reliable News"]]) // 6

    parser = argparse.ArgumentParser(description="Script with --runid argument")
    parser.add_argument('--runid', type=int, choices=range(1, 7), help="Specify a number between 1 and 6")
    args = parser.parse_args()

    if args.runid is None:
        parser.print_help()
        exit(1)

    runid = args.runid

    print(f"Reading the split dataset ../dataset/{runid}_train.csv")
    split_dataset = pd.read_csv(f"../dataset/{runid}_train.csv")

    print(
        f"Need to generate {no_of_reliable_to_generate_per_person} reliable texts & {no_of_hoax_to_generate_per_person} hoax texts"
    )

    hoax_generations = []
    reliable_generations = []

    augment(runid, split_dataset, no_of_hoax_to_generate_per_person,
            no_of_reliable_to_generate_per_person, hoax_generations, reliable_generations)
