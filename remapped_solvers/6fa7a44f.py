def reverse_string(s: str) -> str:
    return s[::-1]

def mirror_string(s: str) -> str:
    reversed_part = reverse_string(s)
    mirrored_string = s + reversed_part
    return mirrored_string
p = lambda input_string: mirror_string(input_string)
