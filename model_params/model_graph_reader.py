from typing import Dict, Tuple, List
import torch
from torch.fx import symbolic_trace

from constants import TORCH_ACTIVATION_FUNCTIONS


def quant_nodes(model: torch.nn.Module) -> Tuple[List[str], List[str]]:
    """
    Identifies nodes in the model where activation quantization should be disabled and determines output nodes.

    Args:
        model: A PyTorch model.

    Returns:
        A tuple containing:
            - A list of node names to disable activation quantization.
            - A list of output node names.
    """
    try:
        nodes_to_disable_activation_quantization, output_nodes = quantization_information(model)
    except:
        nodes_to_disable_activation_quantization, output_nodes = [], []
        for name, module in model.named_children():
            if len(list(module.named_modules())) == 1:
                output_nodes += [name]
            else:
                _nodes_to_disable_activation_quantization, _output_nodes = quant_nodes(module)
                nodes_to_disable_activation_quantization += [name + '.' + _name for _name in _nodes_to_disable_activation_quantization]
                output_nodes += [name + '.' + _name for _name in _output_nodes]
    output_nodes = remove_nodes_with_strs(output_nodes, ['backbone'])
    return nodes_to_disable_activation_quantization, output_nodes

def generate_module_dict(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """
    Creates a dictionary from the PyTorch model's named modules by module name.

    Args:
        model: A PyTorch model.

    Returns:
        A dictionary of the PyTorch model's named modules.
    """
    module_dict = dict()
    for name, module in model.named_modules():
        module_dict[name] = module
    return module_dict


def get_input_names_from_name_list(node, names_list: List[str]) -> List[str]:
    """
    Retrieves input node names from a given node, excluding those in the provided names list.

    Args:
        node: A node from the FX graph.
        names_list: A list of node names to exclude.

    Returns:
        A list of input node target names that are strings.
    """
    input_nodes = node.all_input_nodes
    for in_node in input_nodes:
        if in_node.name not in names_list and in_node.target not in names_list:
            input_nodes += in_node.all_input_nodes
    return [in_node.target for in_node in input_nodes if isinstance(in_node.target, str)]


def remove_nodes_with_strs(str_list: List[str], strs_to_remove: List[str]) -> List[str]:
    """
    Removes strings from the list that contain any of the specified substrings.

    Args:
        str_list: The original list of strings.
        strs_to_remove: Substrings to identify which strings to remove.

    Returns:
        A new list with the specified strings removed.
    """
    new_list = str_list.copy()
    for r in strs_to_remove:
        for s in str_list:
            if r in s:
                new_list.remove(s)
    return new_list

def remove_char_from_str_beginning(str_list: List[str], char: str) -> List[str]:
    """
    Removes a specified character from the beginning of each string in the list if present.

    Args:
        str_list: The original list of strings.
        char: The character to remove from the beginning of each string.

    Returns:
        A new list with the specified character removed from the beginning of each string.
    """
    new_list = []
    for s in str_list:
        if s.startswith(char):
            new_list.append(s[1:])
        else:
            new_list.append(s)
    return new_list


def quantization_information(model: torch.nn.Module) -> Tuple[List[str], List[str]]:
    """
    Gathers quantization-related information from the model's FX graph, identifying nodes where quantization should be disabled.

    Args:
        model: A PyTorch FX model.

    Returns:
        A tuple containing:
            - A list of node names to disable activation quantization.
            - A list of output node names.
    """
    fx_model = symbolic_trace(model)
    module_dict = generate_module_dict(model)
    name_list = list(module_dict.keys())
    # init function variables:
    output_nodes = []
    nodes_to_disable_activation_quantization = []
    for node in fx_model.graph.nodes:
        if node.target in name_list:
            node_module = module_dict[node.target]
            node_type = type(node_module)
            if node_type in TORCH_ACTIVATION_FUNCTIONS:
                nodes_to_disable_activation_quantization += get_input_names_from_name_list(node, name_list)
        elif node.op == 'call_function':
            node_type = node.target
            if node_type in TORCH_ACTIVATION_FUNCTIONS:
                nodes_to_disable_activation_quantization += get_input_names_from_name_list(node, name_list)
        elif node.op == 'placeholder':
            pass
            # nodes_to_disable_activation_quantization += [node.name]
        elif node.op == 'output':
            pass
        elif node.op == 'call_method':
            if hasattr(torch, node.target):
                node_type = getattr(torch, node.target)
            elif hasattr(torch.Tensor, node.target):
                node_type = getattr(torch.Tensor, node.target)
            else:
                raise Exception(f'Call method of type \'{node.target}\' is currently not supported.')
        elif node.op == 'get_attr':
            pass
        else:
            pass

        if node.op == 'output':
            output_nodes += [in_node.name for in_node in node.all_input_nodes]

    nodes_to_disable_activation_quantization = remove_char_from_str_beginning(nodes_to_disable_activation_quantization, '_')
    output_nodes = remove_char_from_str_beginning(output_nodes, '_')
    return nodes_to_disable_activation_quantization, output_nodes