import fire

def remove_blank_lines(output_path):
    input_path = "./dante.txt"
    with open(input_path, 'r') as input_text, open(output_path, 'w') as output_text:
        for line in input_text:
            if line.strip():
                output_text.write(line)


if __name__ == "__main__":
    fire.Fire(remove_blank_lines)

