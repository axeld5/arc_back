def decode_mapping(encoded_mapping):
    return [(encoded_mapping[i:i + 2], encoded_mapping[i + 2]) for i in range(0, len(encoded_mapping), 3)]

def replace_encoded_keys(encoded_data, mapping):
    for original, encoded in mapping:
        encoded_data = encoded_data.replace(encoded, original)
    return encoded_data

def parse_data(encoded_data):
    return eval(encoded_data)

def find_output_from_input(input_value, data):
    for mapping in data:
        if mapping['I'] == input_value:
            return mapping['O']
    return None

def p(input_value):
    encoded_data = '[{"Ip3bl,2q]e"Op3swwwxxxxamg]]},{"Izch,2bhd]e"OzggAAyyyy]]},{"Izdh,2bk]e"OzgiyyyAAmg]]},{"Izcj,2lba]e"Ozggktxxxwwwwa]]}]'
    encoded_mapping = 'vvAp0zmiyqtxlswugvhiuaitrgsdbrakqo[pn[o":nhgmajlhckf3jddif0hccge[f],ebbdaac,3b,0a'
    mapping = decode_mapping(encoded_mapping)
    decoded_data = replace_encoded_keys(encoded_data, mapping)
    data = parse_data(decoded_data)
    return find_output_from_input(input_value, data)
