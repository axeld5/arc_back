def reverse_first_half(concatenated_list):
    first_half = concatenated_list[:5]
    return first_half + first_half[::-1]

def p(input_list):
    return reverse_first_half(input_list)
