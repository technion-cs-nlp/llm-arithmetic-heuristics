import torch
import random
import transformer_lens as lens
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from general_utils import predict_answer
from model_analysis_consts import LLAMA3_8B_CONSTS


OPERATORS = ['+', '-', '*' , '/']
OPERATOR_NAMES = ['addition', 'subtraction', 'multiplication', 'division']
POSITIONS = [1, 2, 3, 4] # All actual token positions (1 = op1, 2 = operator, 3 = op2, 4 = equals sign)
         

def generate_prompts(model: lens.HookedTransformer, 
                    operand_ranges: Dict[str, Tuple[int, int]],
                    validate_numerals: bool = True,
                    correct_prompts: bool = True, 
                    num_prompts_per_operator: Optional[int] = 50,
                    single_token_number_range: Tuple[int, int] = (0, LLAMA3_8B_CONSTS.max_single_token),
                    additional_shots_per_operator: Dict[str, int] = None
                    ):
    """
    Generate arithmetic prompts of the for "x op y=" (without spaces), where x and y are integers and op is an operator.
    The prompts are filtered according to the arguments.

    Args:
        model (nn.Module): The model used to validate answer correctness.
        operand_ranges (dict): A dictionary of the form {operator: (operand_min, operand_max)}.
        validate_numerals (bool): If True, only prompts with valid numerals are returned (no "weird token" answers).
        correct_prompts (bool): If True, only prompts completed correctly by the model are returned. Otherwise, only prompts completed 
                                incorrectly by the model are returned.
        num_prompts_per_operator (int): The number of prompts to generate per operand. If None, all possible prompts are generated.
    
        Returns:
            List[List[Tuple[str, str]]]: The main list is indexed by the different operators, and each list contains num_prompts_per_operator tuples, 
                where each tuple is of form (prompt, answer).
    """
    prompts_and_answers = []

    # assert num_prompts_per_operator > 0, 'num_prompts_per_operator must be a positive integer'
    for operator in operand_ranges.keys():
        assert single_token_number_range[0] <= operand_ranges[operator][0] <= operand_ranges[operator][1] <= single_token_number_range[1], \
                    f'Invalid operand range for operator {operator}'

    for operator in operand_ranges.keys():
        # Generate all possible prompts for the given operator within the operand limits
        operand_min, operand_max = operand_ranges[operator]
        all_operator_prompts = generate_all_prompts_for_operator(operator, operand_min, operand_max, single_token_number_range)

        if additional_shots_per_operator is not None:
            all_operator_prompts = [f"{additional_shots_per_operator[operator]}{p}" for p in all_operator_prompts]

        # Filter the prompts
        filtered_operator_prompts = filter_generated_prompts(model, all_operator_prompts, validate_numerals, correct_prompts)
        assert len(filtered_operator_prompts) > 0, f'No valid prompts for operator {operator} with given parameters'

        # Take k prompts, while maximizing the number of unique answers (so that during patching experiments, the clean and corrupt answers will be different)
        if num_prompts_per_operator is not None:
            filtered_operator_prompts = _maximize_unique_answers(filtered_operator_prompts, k=num_prompts_per_operator)
            
        prompts_and_answers.append(filtered_operator_prompts)
    return prompts_and_answers


def generate_all_prompts_for_operator(operator: str,
                                      operand_min: int,
                                      operand_max: int,
                                      single_token_number_range: Tuple[int, int]) -> List[str]:
    """
    Generate ALL possible "relevant" prompts for a given operator.
    Prompts are valid if there is no illegal operation (e.g. division by zero, negative result, etc).

    Args:
        operator (str): The operator to generate prompts for.
        operand_min (int): The minimum value for the operands.
        operand_max (int): The maximum value for the operands.
    Return:
        List[str]: A list of all possible relevant prompts for a given operator.
    """
    all_operator_prompts = []
    for operand1 in range(operand_min, operand_max):
        operand_2_range = _get_operand_range(operator, operand1, operand_min, operand_max, single_token_number_range[1])
        for operand2 in operand_2_range:
            prompt = '{x}{op}{y}='.format(x=operand1, op=operator, y=operand2)
            answer = eval(prompt[:-1])
            if single_token_number_range[0] <= answer <= single_token_number_range[1]:
                all_operator_prompts.append(prompt)
    return all_operator_prompts


def separate_prompts_and_answers(prompts_and_answers: List[Tuple[str, str]]):
    """
    Separates a list of (prompt, answer) tuples to two lists - one of prompts and one of answers.
    """
    return [pa[0] for pa in prompts_and_answers], [pa[1] for pa in prompts_and_answers]


def filter_generated_prompts(model: lens.HookedTransformer, 
                             prompts: List[str],
                             validate_numerals: bool = True,
                             correct_prompts: bool = True):
    """
    Filters generated prompts according to the given arguments.

    Args:
        model (nn.Module): The model used to validate answer correctness.
        prompts (List[str]): The prompts to filter.
        validate_numerals (bool): If True, only prompts with valid numerals are returned (no "weird token" answers).
        correct_prompts (bool): If True, only prompts completed correctly by the model are returned. Otherwise, only prompts completed 
                                incorrectly by the model are returned.
    
    Returns:
        List[Tuple[str, str]]: A list of (prompt, answer) tuples.
    """
    all_filtered_prompts_and_answers = []
    dataloader = torch.utils.data.DataLoader(prompts, batch_size=32, shuffle=False)
    for batch in tqdm(dataloader):
        filtered_prompts = batch
        answers = predict_answer(model, batch)

        # Use only prompts with numerical answers by the model
        if validate_numerals:
            numerical_indices = [i for i in range(len(answers)) if _is_number(answers[i])]
            filtered_prompts = [filtered_prompts[i] for i in numerical_indices]
            answers = [answers[i] for i in numerical_indices]
        
        # Use only prompts with correct answers (or incorrect, if `correct_prompts` is False)
        is_correct_answers = [_is_answer_correct(prompt, answer) for prompt, answer in zip(filtered_prompts, answers)]
        filtered_prompts = [filtered_prompts[i] for i in range(len(filtered_prompts)) if (correct_prompts and is_correct_answers[i]) or (not correct_prompts and not is_correct_answers[i])]
        answers = [answers[i] for i in range(len(answers)) if (correct_prompts and is_correct_answers[i]) or (not correct_prompts and not is_correct_answers[i])]
        all_filtered_prompts_and_answers.extend(list(zip(filtered_prompts, answers)))

    return all_filtered_prompts_and_answers


def _maximize_unique_answers(rigorous_prompts_and_answers, k=50):
    """
    Get a subset of prompts and answers with as many unique answers as possible.
    Args:
        rigorous_prompts_and_answers (list of tuples): A list of (prompt, answer) pairs.
        k (int, optional): The desired number of prompt-answer pairs in the output list. Defaults to 50.
                           If there are less than k unique answers in the input list, there will be answer repetitions.
    Returns:
        list of tuples: A list of (prompt, answer) pairs with as many unique answers as possible, up to length `k`.
    """
    if len(rigorous_prompts_and_answers) < k:
        new_prompts_and_answers = rigorous_prompts_and_answers + random.choices(rigorous_prompts_and_answers, k=k-len(rigorous_prompts_and_answers))
        random.shuffle(new_prompts_and_answers)
        return new_prompts_and_answers
    else:
        unique_answers = set()
        new_prompts_and_answers = []
        random.shuffle(rigorous_prompts_and_answers)
        for prompt, answer in rigorous_prompts_and_answers:
            if answer not in unique_answers:
                unique_answers.add(answer)
                new_prompts_and_answers.append((prompt, answer))
        if len(new_prompts_and_answers) < k:
            new_prompts_and_answers += random.choices(rigorous_prompts_and_answers, k=k-len(new_prompts_and_answers))
            
        return new_prompts_and_answers[:k]


def _get_operand_range(operator, previous_operand, operand_min, operand_max, max_single_token_value):
    if operator == '+':
        return range(operand_min, min(max_single_token_value - previous_operand, operand_max))
    elif operator == '-':
        return range(operand_min, previous_operand + 1)
    elif operator == '*':
        if previous_operand == 0:
            return range(operand_min, operand_max)
        else:
            return range(operand_min, min((max_single_token_value // previous_operand) + 1, operand_max))
    elif operator == '/':
        return range(max(1, operand_min), operand_max)
    else:
        raise ValueError(f'Operator {operator} is not supported')



def _is_answer_correct(prompt: str, answer: str, convert_to_int: bool = True):
    """
    Checks if an answer is a correct completion to a prompt.
    Whitespaces are ignored.

    Args:
        prompt (str): The prompt (for example '5+4=')
        answer (str): The answer (for example '9')
        convert_to_int (bool): If True, the ground truth answer is converted to an integer before comparison to the tested answer.
    """
    # Handle few-shot case
    few_shot_sep = ';' if ';' in prompt else (',' if ',' in prompt else None)
    if few_shot_sep is not None:
        prompt = prompt[prompt.rfind(few_shot_sep) + 1:]

    real_answer = eval(prompt.replace('=', ''))
    if convert_to_int:
        real_answer = int(real_answer)
    try:
        return real_answer == _to_number(answer)
    except ValueError:
        return False


def _to_number(s: str):
    try:
        return int(s)
    except ValueError:
        return float(s)


def _is_number(s: str, is_int=False):
    try:
        if is_int:
            int(s)
        else:
            float(s)
        return True
    except ValueError:
        return False
    

def is_writing_of_number(s: str):
    word_to_number = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'million': 1000000
    }

    words = s.split()
    for word in words:
        if word not in word_to_number:
            return False
    return True