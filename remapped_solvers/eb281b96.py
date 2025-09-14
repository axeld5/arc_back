def create_palindrome_segment(segment):
    return segment + segment[-2:0:-1]

def extend_with_first_element(palindrome_segment, first_element):
    return palindrome_segment * 2 + first_element

def create_extended_palindrome(sequence):
    palindrome_segment = create_palindrome_segment(sequence)
    first_element = sequence[:1]
    return extend_with_first_element(palindrome_segment, first_element)

def p(sequence):
    return create_extended_palindrome(sequence)
