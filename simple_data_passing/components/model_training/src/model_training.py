import fire


def count_lines(input_data, output_path):
    file = open(input_data, "r")

    line_count = 0

    for line in file:
        if line != "\n":
            line_count += 1
    file.close()

    with open(output_path, 'w') as output_text:
        output_text.write(str(line_count))


if __name__ == "__main__":
    fire.Fire(count_lines)

