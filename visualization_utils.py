import transformer_lens as lens
import plotly.express as px
import plotly.graph_objects as go
import circuitsvis as cv
import torch
from typing import List
from component import Component


def imshow(tensor, **kwargs):
    px.imshow(lens.utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show()


def line(tensor, **kwargs):
    px.line(y=tensor, **kwargs).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    px.scatter(y=lens.utils.to_numpy(y), x=lens.utils.to_numpy(x), labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs).show()


def scatter_with_labels(x, y, hovertext, color=None, mode='markers', **layout_kwargs):
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode=mode, hovertext=hovertext, marker=dict(color=color)))
    fig.update_layout(**layout_kwargs).show()


def multiple_lines(x, y, line_titles, add_vlines_at_maximum=False, show_fig=True, hovertext=None, colors=None, **layout_kwargs):
    traces = []
    colors = colors or px.colors.qualitative.Plotly
    for i in range(len(line_titles)):
        trace = go.Scatter(x=x, y=y[i], mode='lines', name=line_titles[i], hovertext=hovertext, line=dict(color=colors[i % len(colors)]))
        traces.append(trace)

    fig = go.Figure(traces)

    if add_vlines_at_maximum:
        for i, trace in enumerate(traces):
            fig.add_vline(x[y[i].argmax()], line_dash="dash", line_color=colors[i])

    fig.update_layout(**layout_kwargs)

    if show_fig:
        fig.show()
    else:
        return fig


def visualize_arithmetic_attention_patterns(model: lens.HookedTransformer, 
                                            components: List[Component], 
                                            prompts: List[str],
                                            use_bos_token: bool = True, 
                                            return_raw_patterns: bool = False):
    """
    Visualize the resulting attention patterns for a list of attention heads, averaged across a list of prompts.

    Args:
        model (lens.HookedTransformer): The model to visualize.
        components (List[Component]): The attention heads to visualize. If any components are not attention heads, they are ignored.
        prompts (List[str]): The prompts to pass through the model and average over.
        use_bos_token (bool): Should the BOS token be part of the prompt passed through the model.
        return_raw_patterns (bool): If True, the raw activation in the pattern hook are also returned.
    Returns:
        (circuitsvis.utils.render.RenderedHTML) - The rendered HTML visualization.
    """
    prompt_loader = torch.utils.data.DataLoader(prompts, batch_size=32, shuffle=False)

    labels = [f'{head_component.layer}H{head_component.head_idx}' for head_component in components]
    patterns = [[] for _ in range(len(components))]
    for batch in prompt_loader:
        _, cache = model.run_with_cache(batch, return_type='logits', prepend_bos=use_bos_token)

        for i, head_component in enumerate(components):
            if head_component.head_idx is None:
                # Ignore non-head components (mlp etc)
                continue
            patterns[i].append(cache['pattern', head_component.layer].cpu()[:, head_component.head_idx])
        del cache

    patterns = [torch.cat(p).mean(dim=0) for p in patterns] # Unify batches and mean across prompts to single tensor
    patterns = torch.stack(patterns, dim=0) # Convert list to a single tensor for visualization

    # Get the axis labels. In case the prompts are averaged
    if len(prompts) == 1:
        str_tokens = model.to_str_tokens(prompts[0], prepend_bos=use_bos_token)
    else:
        str_tokens = ['operand1', 'operator', 'operand2', '=']
        if use_bos_token:
            str_tokens.insert(0, 'BOS')

    heads_html = cv.attention.attention_heads(attention=patterns, tokens=str_tokens, attention_head_names=labels)
    if return_raw_patterns:
        return heads_html, patterns
    else:
        return heads_html
