def split_list(input_list, slice_size):
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def create_sentence_chunks(pages_and_texts, num_sentence_chunk_size):
    for item in pages_and_texts:
        item["sentence_chunks"] = split_list(
            input_list=item["sentences"],
            slice_size=num_sentence_chunk_size
        )
        item["num_chunks"] = len(item["sentence_chunks"])

    return pages_and_texts
