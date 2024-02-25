from Layers.ELMLayer import ELMLayer
from Layers.KELMLayer import KELMLayer
from Layers.WELMLayer import WELMLayer
from Layers.SubELMLayer import SubELMLayer


def get_layers(input_dict):
    """
        Create a list of layer instances based on the input dictionary.

        This function parses a dictionary containing layer configurations and creates
        layer instances accordingly.

        Parameters:
        -----------
            input_dict (dict): A dictionary containing layer configurations.

        Returns:
        -----------
            list: A list of layer instances created based on the input dictionary.
    """
    result_list = []
    sub_dict = {}

    i = 0
    # Iterate through each key-value pair in the input dictionary
    for key, value in input_dict.items():
        # Split the key into parts using '_' as a delimiter
        key_parts = key.split('.')

        sub_dict.update({key_parts[-1]: value})
        # Check if the key follows the 'layer_n_something' pattern
        if key_parts[-1] == 'C' and i != 0:
            result_list.append(sub_dict)
            sub_dict = {key_parts[-1]: value}
        i += 1
    result_list.append(sub_dict)

    layers = []
    for layer in result_list:
        if 'name' in layer:
            n = layer.pop('name')
        e = eval(f'{n}(**layer)')
        layers.append(e)
    return layers