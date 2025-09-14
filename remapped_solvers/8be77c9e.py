def reverse_and_concatenate(original_string):
    reversed_string = original_string[::-1]
    concatenated_result = original_string + reversed_string
    return concatenated_result

def p(input_string):
    return reverse_and_concatenate(input_string)
