from component import Component
import torch


class Circuit():
    """
    A class representing a circuit in a transformer model.
    A circuit is a set of Component objects, which are connected as a DAG. Each connection represents the 
    effect of an early component on a late component.
    """
    def __init__(self, model_cfg) -> None:
        """
        Initilize an empty circuit, as part of a transformer model.

        Args:
            model_cfg (lens.HookedTransformerConfig): The configuration of the transformer model.
        """
        # Each key is a (early_component, late_component) tuple, and each value represents the effect of patching the early component 
        # on the late component (via path patching). In case the late component is a psuedo "logits" component, the value is the effect
        # of activation patching the early component.
        self.components = set()
        self.edges = {}
        self.model_cfg = model_cfg

    def add_component(self, component, patching_effects=None) -> None:
        """
        Add a component to the circuit, and connect it to the logits affected by it / to other components which affect it.

        Args:
            component (Component): The component to add.   
            patching_effects (torch.Tensor): This is a matrix of the effects of early components on the given component (via path patching). 
                The tensor must be of shape (num_layers, 1) in case the early components are only MLPs, 
                (num_layers, num_heads) in case the early components are only z vectors,
                or (num_layers, num_heads + 1) in case the early components are both z vectors and MLPs (the MLP should be the last column).   
                If None, the component is added without any edges to it.                          
        """
        # The effects is a matrix representing the effect of many early components (in a certain structure) on the late component
        assert patching_effects is None or len(patching_effects.shape) == 2

        self.components.add(component)

        if patching_effects is not None:
            layer_count, component_count = patching_effects.shape 
            assert layer_count == self.model_cfg.n_layers, \
                'The number of rows in the patching effects matrix must be equal to the number of layers in the model.'
            
            mlp_column_count = 1
            z_column_count = self.model_cfg.n_heads
            assert component_count in [mlp_column_count, z_column_count, z_column_count + mlp_column_count], \
                f'Invalid number of columns in the patching effects matrix (got {component_count}, expected one of ' \
                f'{[mlp_column_count, z_column_count, z_column_count + mlp_column_count]}'
            
            if component_count == mlp_column_count or component_count == (z_column_count + mlp_column_count):
                # Patching effects include effects of MLPs on component (in the last column)
                for layer in range(layer_count):
                    early_component = Component('mlp_out', layer=layer)
                    self.edges[(early_component, component)] = patching_effects[layer, -1]

            if component_count == z_column_count or component_count == (z_column_count + mlp_column_count):
                for layer in range(layer_count):
                    for head_idx in range(z_column_count):
                        early_component = Component('z', head=head_idx, layer=layer)
                        self.edges[(early_component, component)] = patching_effects[layer, head_idx]

    def remove_component(self, component) -> None:
        self.components.remove(component)
        for edge in list(self.edges.keys()):
            if edge[1] == component:
                del self.edges[edge]
                
    def get_component_patching_effects(self, component, include_attn=True, include_mlp=True, is_component_late=True, zero_non_existing_edges=False) -> float:
        """
        Get the patching effects of all early components on a given component / the effects of the given component on later components.

        Args:
            component (Component): The component to get the patching effects for.
            include_attn (bool): If True, the patching effects of attention heads are included.
            include_mlp (bool): If True, the patching effects of MLPs are included.
            is_component_late (bool): If True, the component is considered a late component, and the patching effects of early components on it are returned.
                If False, the component is considered an early component, and the patching effects of it on later components are returned.
            zero_non_existing_edges (bool): If True, the patching effects of non-existing edges are returned as 0. Otherwise, an exception is raised.
        Returns:
            (torch.tensor) - A tensor of shape (num_layers, num_components_per_layer) representing the patching effects of early components on the given component.
                             The num of components per layer is determined by the flags include_attn and include_mlp (it can be one of 1, n_heads or n_heads + 1).
        """
        assert include_attn or include_mlp, 'You must request at least one of the patching effects (attn heads or mlp) on the component'
        patching_effect_columns = (self.model_cfg.n_heads if include_attn else 0) + (1 if include_mlp else 0)
        effect_matrix = torch.zeros((self.model_cfg.n_layers, patching_effect_columns))
        for layer in range(self.model_cfg.n_layers):
            if include_attn:
                for head_idx in range(self.model_cfg.n_heads):
                    if is_component_late:
                        head_component = Component('z', head=head_idx, layer=layer)
                        effect_matrix[layer, head_idx] = self.edges.get((head_component, component), 0 if zero_non_existing_edges else None)
                    else:
                        # Edges are saved such that q_input/k_input/v_input are the late components, so searching for a late component with name=z doesnt work.
                        effect_matrix[layer, head_idx] = self.edges.get((component, Component('q_input', head=head_idx, layer=layer)), 0 if zero_non_existing_edges else None) + \
                            self.edges.get((component, Component('k_input', head=head_idx, layer=layer)), 0 if zero_non_existing_edges else None) + \
                            self.edges.get((component, Component('v_input', head=head_idx, layer=layer)), 0 if zero_non_existing_edges else None)
            if include_mlp:
                if is_component_late:
                    mlp_component = Component('mlp_out', layer=layer)
                    effect_matrix[layer, -1] = self.edges.get((mlp_component, component), 0 if zero_non_existing_edges else None)
                else:
                    mlp_component = Component('mlp_in', layer=layer)
                    effect_matrix[layer, -1] = self.edges.get((component, mlp_component), 0 if zero_non_existing_edges else None)
        return effect_matrix
