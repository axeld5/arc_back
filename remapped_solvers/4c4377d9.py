def reverse_and_concatenate(input_string):
    reversed_string = input_string[::-1]
    concatenated_result = reversed_string + input_string
    return concatenated_result

def p(input_string):
    return reverse_and_concatenate(input_string)
