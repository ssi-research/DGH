import copy
from typing import Type, List, Dict, Tuple, Optional
import torch
from torch import Tensor
from torch.fx import symbolic_trace, GraphModule
from torch.nn import Module, Linear, Conv2d
from constants import DEVICE, IMAGE_INPUT
from model_params.model_graph_reader import quant_nodes


class OrigBNStatsHolder(object):
    """
    Holds the original batch normalization (BN) statistics for a model.
    """
    def __init__(self,
                 model: Module,
                 bn_layer_types: Type[list],
                 eps=1e-6)-> None:
        """
        Initializes the OrigBNStatsHolder.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (List[Type[Module]]): List of batch normalization layer types.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.
        """
        self.bn_params = self.get_bn_params(model, bn_layer_types)
        self.eps = eps

    def get_bn_layer_names(self)-> List[str]:
        """
        Get the names of all batch normalization layers.

        Returns:
            List[str]: List of batch normalization layer names.
        """
        return list(self.bn_params.keys())

    def get_mean(self, bn_layer_name: str)-> Tensor:
        """
        Get the mean of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Mean of the batch normalization layer.
        """
        return self.bn_params[bn_layer_name][0]

    def get_var(self, bn_layer_name: str)-> Tensor:
        """
        Get the variance of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Variance of the batch normalization layer.
        """
        return self.bn_params[bn_layer_name][1]

    def get_std(self, bn_layer_name: str)-> Tensor:
        """
        Get the standard deviation of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Standard deviation of the batch normalization layer.
        """
        var = self.bn_params[bn_layer_name][1]
        eps = self.eps
        std = torch.sqrt(var + eps)
        while torch.any(torch.isnan(std)):
            eps *= 10
            std = torch.sqrt(var + eps)
        return std

    def get_num_bn_layers(self) -> int:
        """
        Get the number of batch normalization layers.

        Returns:
            int: Number of batch normalization layers.
        """
        return len(self.bn_params)

    @staticmethod
    def get_bn_params(model: Module, bn_layer_types: List[Type[Module]]) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Get the batch normalization parameters (mean and variance) for each batch normalization layer in the model.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (List[Type[Module]]): List of batch normalization layer types.

        Returns:
            Dict[str, Tuple[Tensor, Tensor]]: Dictionary mapping batch normalization layer names to their parameters.
        """
        bn_params = {}
        # Assume the images in the dataset are normalized to be 0-mean, 1-variance
        imgs_mean = torch.zeros(1, 3).to(DEVICE)
        imgs_var = torch.ones(1, 3).to(DEVICE)
        bn_params.update({IMAGE_INPUT: (imgs_mean, imgs_var)})
        for name, module in model.named_modules():
            if isinstance(module, tuple(bn_layer_types)):
                mean = module.running_mean.detach().clone().flatten().to(DEVICE)
                var = module.running_var.detach().clone().flatten().to(DEVICE)
                bn_params.update({name: (mean, var)})
        return bn_params


class InputHook(object):
    """
    Forward hook used to extract the input of an intermediate batch norm layer.
    """
    def __init__(self) -> None:
        """
        Initialize the InputHook.
        """
        super(InputHook, self).__init__()
        self.input = None

    def hook(self, module: Module, input: Tuple[Tensor, ...], output: Tensor) -> None:
        """
        Hook function to extract the input of the batch normalization layer.

        Args:
            module (Module): Batch normalization module.
            input (Tuple[Tensor, ...]): Input tensor(s).
            output (Tensor): Output tensor.
        """
        self.input = input[0]

    def clear(self) -> None:
        """
        Clear the stored input tensor.
        """
        self.input = None


class OutputHook(object):
    """
    Forward hook used to extract the output of an intermediate batch norm layer.
    """
    def __init__(self) -> None:
        """
        Initialize the OutputHook.
        """
        super(OutputHook, self).__init__()
        self.output: Optional[Tensor] = None

    def hook(self, module: Module, input: Tuple[Tensor, ...], output: Tensor) -> None:
        """
        Hook function to extract the output of the batch normalization layer.

        Args:
            module (Module): Batch normalization module.
            input (Tuple[Tensor, ...]): Input tensor(s).
            output (Tensor): Output tensor.
        """
        self.output = output

    def clear(self) -> None:
        """
        Clear the stored output tensor.
        """
        self.output = None


class ActivationExtractor(object):
    """
    Extracts activations of inputs to batch normalization layers in a model.
    """
    def __init__(
        self,
        model: Module,
        bn_layer_types: List[Type[Module]]
    ) -> None:
        """
        Initializes the ActivationExtractor.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (List[Type[Module]]): List of batch normalization layer types.
        """
        self.model = copy.deepcopy(model)
        _, _output_nodes = quant_nodes(self.model)
        self._output_nodes = [o.replace('.', '_') for o in _output_nodes]
        self.bn_layer_types = bn_layer_types
        # self.model = model
        self.num_bn_layers = sum([1 if isinstance(layer, tuple(bn_layer_types)) else 0 for layer in model.modules()])
        print(f'Number of BatchNorm layers = {self.num_bn_layers}')

        self.hooks = {}  # Dictionary to store InputHook instances by layer name
        self.last_linear_layers_input_hooks = {}  # Dictionary to store InputHook instances by layer name
        self.last_linear_layers_output_hooks = {}  # Dictionary to store OutputHook instances by layer name
        self.grad_hooks_input = {}
        self.grad_hooks_output = {}
        self.hook_handles = []  # List to store hook handles
        self.last_linear_layer_weights = []  # list of the last linear layers' weights

        # set hooks for batch norm layers
        self._set_hooks_for_layers()

        # set hooks for last output layers
        self._set_hooks_for_last_layers()

    def to_fx_by_modules(self, model: Module) -> List[GraphModule]:
        """
        Converts the model to FX GraphModules by traversing its modules recursively.

        Args:
            model (Module): The PyTorch model.

        Returns:
            List[GraphModule]: List of FX GraphModules derived from the model's modules.
        """
        try:
            fx_graphs = [symbolic_trace(model)]
        except:
            if len(list(model.named_modules())) == 1:
                try:
                    return [symbolic_trace(model)]
                except:
                    return []
            fx_graphs = []
            for name, module in model.named_children():
                print(name)
                fx_graphs += self.to_fx_by_modules(module)
        return fx_graphs

    def _set_hooks_for_layers(self) -> None:
        """
        Sets forward hooks for the inputs of all batch normalization layers.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, self.bn_layer_types):
                hook = InputHook()  # Create an InputHook instance
                self.hooks.update({name: hook})
                hook_handle = module.register_forward_hook(hook.hook)  # Register the hook on the module
                self.hook_handles.append(hook_handle)  # Store the hook handle in the hook_handles list

    def _set_hooks_for_last_layers(self) -> None:
        """
        Sets forward hooks for the input and output of the last linear or convolutional layers in the model.
        """

        for name, module in self.model.named_modules():
            if name.replace('.', '_') in self._output_nodes:
                if isinstance(module, Linear) or isinstance(module, Conv2d):
                    self.last_linear_layer_weights.append(module.weight.data.clone())
                    input_hook = InputHook()  # Create an InputHook instance
                    output_hook = OutputHook()  # Create an OutputHook instance
                    self.last_linear_layers_input_hooks.update({name: input_hook})
                    self.last_linear_layers_output_hooks.update({name: output_hook})
                    input_hook_handle = module.register_forward_hook(
                        input_hook.hook)
                    output_hook_handle = module.register_forward_hook(
                        output_hook.hook)  # Register the hook on the module
                    self.hook_handles.append(input_hook_handle)
                    self.hook_handles.append(output_hook_handle)

    def get_grads_outputs(self, layer_name: str) -> Tensor:
        """
        Retrieves the gradients of the outputs for a specific layer.

        Args:
            layer_name (str): Name of the layer.

        Returns:
            Tensor: Gradients of the outputs for the specified layer.
        """
        return self.grad_hooks_output.get(layer_name).output

    def get_grads_inputs(self, layer_name: str) -> Tensor:
        """
        Retrieves the gradients of the inputs for a specific layer.

        Args:
            layer_name (str): Name of the layer.

        Returns:
            Tensor: Gradients of the inputs for the specified layer.
        """
        return self.grad_hooks_input.get(layer_name).input

    def get_activation(self, layer_name: str) -> Tensor:
        """
        Get the activation (input) tensor of a batch normalization layer.

        Args:
            layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Activation tensor of the batch normalization layer.
        """
        return self.hooks.get(layer_name).input

    def get_output_layer_input_activation(self) -> List[Tensor]:
        """
        Get the input activation tensors of the output layers.

        Returns:
            List[Tensor]: List of input activation tensors of the output layers.
        """
        return [v.input for v in self.last_linear_layers_input_hooks.values()]

    def get_output_layer_output_activation(self) -> List[Tensor]:
        """
        Get the output activation tensors of the output layers.

        Returns:
            List[Tensor]: List of output activation tensors of the output layers.
        """
        return [v.output for v in self.last_linear_layers_output_hooks.values()]

    def get_num_bn_layers(self) -> int:
        """
        Get the number of batch normalization layers.

        Returns:
            int: Number of batch normalization layers.
        """
        return self.num_bn_layers

    def get_bn_layer_names(self) -> List[str]:
        """
        Get a list of batch normalization layer names.

        Returns:
            List[str]: A list of batch normalization layer names.
        """
        return list(self.hooks.keys())

    def clear(self) -> None:
        """
        Clear the stored activation tensors.
        """
        for hook in self.hooks.values():
            hook.clear()
        for hook in self.last_linear_layers_input_hooks.values():
            hook.clear()

    def remove(self) -> None:
        """
        Remove the hooks from the model.
        """
        self.clear()
        for handle in self.hook_handles:
            handle.remove()

    def run_on_inputs(self, inputs: Tensor) -> Tensor:
        """
        Runs the model on the given inputs and returns the output.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(inputs)