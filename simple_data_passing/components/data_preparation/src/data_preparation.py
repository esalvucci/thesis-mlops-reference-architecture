import fire
import requests


def remove_blank_lines(input_dataset_url, output_path):
    
    response = requests.get(input_dataset_url)
    input_filename = 'dante.txt'

    with open(input_filename, mode='wb') as localfile:
        localfile.write(response.content)

    with open(input_filename, 'r') as input_text, open(output_path, 'w') as output_text:
        for line in input_text:
            if line.strip():
                output_text.write(line)


if __name__ == "__main__":
    fire.Fire(remove_blank_lines)

