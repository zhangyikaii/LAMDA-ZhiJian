from zhijian.models.utils import get_class_from_module
import zhijian.models.addin.module as addin_module

import re
import pkgutil
import inspect
from collections import defaultdict


def prepare_specific_addin_parser(parser):
    return parser

def prepare_addins(args, model_args, addin_classes=[], **kwargs):
    addins, fixed_params = [], []
    addin_name2class = {i.__name__:i for i in addin_classes}

    for cur_addin in args.addins:
        addin_name = cur_addin['name']
        cur_kwargs = {**kwargs, **cur_addin['kwargs']} if 'kwargs' in cur_addin.keys() else kwargs
        if addin_name in addin_name2class.keys():
            cur_addin_class = addin_name2class[addin_name]
        else:
            cur_addin_class = get_class_from_module(f'zhijian.models.addin.module.{addin_name.lower()}', addin_name)
        addins.append(cur_addin_class(args, model_args, **cur_kwargs))

    return addins, fixed_params

def _compile_error():
    raise NotImplementedError

def module_compile(input_str):
    matches = re.finditer(r'\[(.*?)\]|([^\[\].]+)', input_str)

    results = []
    for match in matches:
        if match.group(1):
            results.append(int(match.group(1)))
        else:
            results.append(match.group(2))
    if not results:
        results.append(input_str)
    return results

def index_range_compile(input_str):
    results = []
    pattern = r'\[(\d+):(\d+)\]'
    matches = re.findall(pattern, input_str)
    if not matches:
        return [input_str]

    matches = list(set(matches))
    if len(matches) > 1:
        differences = [int(match[1]) - int(match[0]) for match in matches]
        is_equal = all(diff == differences[0] for diff in differences)
        if not is_equal:
            _compile_error()

    index_range_length = int(matches[0][1]) - int(matches[0][0])

    for index_incre in range(index_range_length):
        cur_str = input_str
        for match in matches:
            cur_str = cur_str.replace(f'{match[0]}:{match[1]}', f'{int(match[0]) + index_incre}')
        results.append(cur_str)

    return results

def reuse_keys_config_compile(input_str):
    reuse_keys = []
    input_str_copied = index_range_compile(input_str)
    for cur_input_str in input_str_copied:
        reuse_keys.append(module_compile(cur_input_str))
    
    return reuse_keys

def addin_config_compile(input_str):
    addin_classes = set()
    module_path = addin_module.__path__[0]
    for _, cur_module_name, _ in pkgutil.iter_modules([module_path]):
        cur_addin_module = __import__(f'zhijian.models.addin.module.{cur_module_name}', fromlist=[cur_module_name])
        addin_classes.update([i for i, _ in inspect.getmembers(cur_addin_module, inspect.isclass)])

    def _get_item(l, x):
        if x < 0 or x >= len(l):
            return None
        return l[x]
    def _is_block(x):
        if x is None or 'block_value' not in x.keys():
            return False
        return True

    input_str_copied = index_range_compile(input_str)

    addins = []
    for cur_input_str in input_str_copied:
        sub_input_str_1, sub_input_str_2 = cur_input_str.split(':')[0].strip(), cur_input_str.split(':')[1].strip()

        match_first_bracket = re.search(r'\((.*?)\)', sub_input_str_1)
        if not match_first_bracket:
            _compile_error()

        cur_addin = module_compile(match_first_bracket.group(1))
    
        matches = re.finditer(r'\((.*?)\)|\{(.*?)\}|->|...', sub_input_str_2)

        elements, in_positions, out_positions, inout_positions = [], [], [], []
        addin2pos, pos2element = defaultdict(str), defaultdict(int)
        for i_match, match in enumerate(matches):

            cur_kwargs = {}
            if match.group(1):
                cur_element = match.group(1)
                cur_kwargs = {'block_value': module_compile(cur_element)}
            elif match.group(2):
                cur_element = match.group(2)
                cur_kwargs = {
                    'block_value': module_compile(cur_element)
                    }
                if cur_element.startswith('inout'):
                    inout_positions.append(i_match)
                elif cur_element.startswith('in'):
                    in_positions.append(i_match)
                elif cur_element.startswith('out'):
                    out_positions.append(i_match)
                else:
                    _compile_error()
            else:
                cur_element = match.group()
            elements.append({
                'value': cur_element,
                'start': match.start(),
                'end': match.end(),
                **cur_kwargs
            })

   
        # inout:
        for cur_inout_pos in inout_positions:
            bef_element, aft_element = _get_item(elements, cur_inout_pos - 1), _get_item(elements, cur_inout_pos + 1)
            if _is_block(bef_element):
                cur_handle_element = bef_element
                cur_hook_pos = 'post'
            elif _is_block(aft_element):
                cur_handle_element = aft_element
                cur_hook_pos = 'pre'
            else:
                _compile_error()

            addins.append({
                'name': cur_addin[0],
                'location': [cur_handle_element['block_value']],
                'hook': [[cur_addin[1], cur_hook_pos]]
            })

        # in and out
        if len(in_positions) != len(out_positions):
            _compile_error()

        out_value2pos = {elements[i]['value']: i for i in out_positions}

        in_and_out_pos = []
        for i in in_positions:
            cur_out_value = elements[i]['value'].replace('in', 'out')
            if cur_out_value in out_value2pos.keys():
                in_and_out_pos.append((i, out_value2pos[cur_out_value]))

        for cur_in_pos, cur_out_pos in in_and_out_pos:
            bef_element_1, aft_element_1 = _get_item(elements, cur_in_pos - 1), _get_item(elements, cur_in_pos + 1)
            bef_element_2, aft_element_2 = _get_item(elements, cur_out_pos - 1), _get_item(elements, cur_out_pos + 1)
            (cur_handle_element_1, cur_hook_pos_1) = (bef_element_1, 'post') if _is_block(bef_element_1) else (aft_element_1, 'pre')
            (cur_handle_element_2, cur_hook_pos_2) = (bef_element_2, 'post') if _is_block(bef_element_2) else (aft_element_2, 'pre')
            if not _is_block(cur_handle_element_1) or not _is_block(cur_handle_element_2):
                _compile_error()
            addins.append({
                'name': cur_addin[0],
                'location': [cur_handle_element_1['block_value'], cur_handle_element_2['block_value']],
                'hook': [[f'get_{cur_hook_pos_1}', cur_hook_pos_1], [cur_addin[1], cur_hook_pos_2]]
            })

    return addins


def addin_config_gui_compile(input_str):
    addin_classes = set()
    module_path = addin_module.__path__[0]
    for _, cur_module_name, _ in pkgutil.iter_modules([module_path]):
        cur_addin_module = __import__(f'zhijian.models.addin.module.{cur_module_name}', fromlist=[cur_module_name])
        addin_classes.update([i for i, _ in inspect.getmembers(cur_addin_module, inspect.isclass)])

    def _check_is_addin(cur_block):
        for i in addin_classes:
            if i in cur_block:
                return True
        return False
    def _compile_error():
        raise NotImplementedError
    def _get_item(l, x):
        if x < 0 or x >= len(l):
            return None
        return l[x]

    input_str_list = [i for i in input_str.splitlines() if len(i.strip()) != 0]
    input_index_range_compiled = {i: index_range_compile(i) for i in input_str_list}
    index_range_length = max([len(i) for i in input_index_range_compiled.values()])
    input_str_copied = []
    for i_copied in range(index_range_length):
        cur_copied = input_str_list.copy()
        for k, v in input_index_range_compiled.items():
            if len(v) not in [1, index_range_length]:
                _compile_error()
            if len(v) == 1:
                continue
            cur_copied[input_str_list.index(k)] = v[i_copied]
        input_str_copied.append(cur_copied)


    for cur_input_str in input_str_copied:
        processed_info, processed_addin2pos, processed_pos2element = [], [], []
        processed_addin_keys = []
        for idx, cur_str in enumerate(cur_input_str):
            matches = re.finditer(r'\((.*?)\)|->|\|', cur_str)

            elements, arrows = [], []
            addin2pos, pos2element = defaultdict(str), defaultdict(int)
            for match in matches:
                if match.group() == '->':
                    cur_element = match.group()
                    arrows.append({
                        'value': cur_element,
                        'compiled_value': module_compile(match.group()),
                        'start': match.start(),
                        'end': match.end(),
                        'is_addin': False
                    })
                else:
                    cur_element = match.group(1) if match.group(1) else match.group()
                    addin2pos[cur_element] = match.start()
                    is_addin = _check_is_addin(cur_element)
                    if is_addin:
                        processed_addin_keys.append([idx, len(elements), cur_element])
                    elements.append({
                        'value': cur_element,
                        'compiled_value': module_compile(cur_element),
                        'start': match.start(),
                        'end': match.end(),
                        'is_addin': is_addin
                    })
                pos2element[match.start()] = cur_element

            if len(elements) - 1 != len(arrows):
                _compile_error()
            for i_check in range(len(arrows)):
                if elements[i_check]['end'] != arrows[i_check]['start'] or elements[i_check + 1]['start'] != arrows[i_check]['end']:
                    _compile_error()

            processed_info.append({'elements': elements, 'arrows': arrows})
            processed_addin2pos.append(addin2pos)
            processed_pos2element.append(pos2element)

        addins = []
        for row_idx, cur_element_idx, cur_addin_name in processed_addin_keys:
            bef_element, aft_element = _get_item(processed_info[row_idx]['elements'], cur_element_idx - 1), _get_item(processed_info[row_idx]['elements'], cur_element_idx + 1)

            cur_element = processed_info[row_idx]['elements'][cur_element_idx]

            if bef_element is None and aft_element is None:
                _compile_error()

            def _find_aligned_connection(aligned_aft_pos):
                aligned_threshold = 5
                for sub_row_idx in range(len(processed_info)):
                    if sub_row_idx == row_idx:
                        continue
                    for sub_element_idx, sub_element in enumerate(processed_info[sub_row_idx]['elements']):
                        if sub_element['value'] == '|' and abs(sub_element['start'] - aligned_aft_pos) <= aligned_threshold:
                            return sub_row_idx, sub_element_idx
                return None, None

            if bef_element is None and aft_element['value'] != '|':
                addins.append({
                    'name': cur_element['compiled_value'][0],
                    'location': [aft_element['compiled_value']],
                    'hook': [[cur_element['compiled_value'][1], 'pre']]
                })
            elif aft_element is None and bef_element['value'] != '|':
                addins.append({
                    'name': cur_element['compiled_value'][0],
                    'location': [bef_element['compiled_value']],
                    'hook': [[cur_element['compiled_value'][1], 'post']]
                })
            elif aft_element['value'] == '|':
                aft_element_pos = aft_element['start']

                aligned_aft_row_idx, aligned_aft_element_idx = _find_aligned_connection(aft_element_pos)
                if aligned_aft_element_idx is None:
                    _compile_error()
                in2_element = _get_item(processed_info[aligned_aft_row_idx]['elements'], aligned_aft_element_idx - 1)

                if bef_element['value'] != '|':
                    addins.append({
                        'name': cur_element['compiled_value'][0],
                        'location': [bef_element['compiled_value'], in2_element['compiled_value']],
                        'hook': [['get_post', 'post'], [cur_element['compiled_value'][1], 'post']]
                    })
                else:
                    bef_element_pos = bef_element['start']
                    aligned_bef_row_idx, aligned_bef_element_idx = _find_aligned_connection(bef_element_pos)

                    if aligned_bef_element_idx is None:
                        _compile_error()
                    in1_element = _get_item(processed_info[aligned_bef_row_idx]['elements'], aligned_bef_element_idx + 1)

                    addins.append({
                        'name': cur_element['compiled_value'][0],
                        'location': [in1_element['compiled_value'], in2_element['compiled_value']],
                        'hook': [['get_pre', 'pre'], [cur_element['compiled_value'][1], 'post']]
                    })

        return addins
