import os
import sys
import re
import json
import subprocess
import shutil
import tempfile
import traceback
import signal
import logging
import stat
import pwd
import grp
import glob
import base64
import string
import platform
import time
import select
import random
import shlex
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional, Union, NamedTuple
from collections import defaultdict, deque, Counter
from functools import reduce, partial
from copy import deepcopy
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Thread, Timer
from contextlib import contextmanager
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import http.server
import socketserver
import atexit
import webbrowser
from clang.cindex import CompilationDatabase, Index, TranslationUnit
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from graphviz import Digraph
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import tiktoken
import chardet
import signal
from contextlib import contextmanager
import networkx as nx
import numpy as np
import tomlkit
from pydantic import BaseModel
import openai
from openai import OpenAI, BadRequestError, AzureOpenAI
import anthropic
from anthropic import AnthropicBedrock, InternalServerError
import replicate
# import google.generativeai as genai
# from google.generativeai.protos import Content, Part
import clang.cindex
clang.cindex.Config.set_library_file('/usr/lib/llvm-19/lib/libclang.so.1')
# clang.cindex.Config.set_library_file('/opt/homebrew/opt/llvm/lib/libclang.dylib')  # for macOS
from intervaltree import Interval, IntervalTree
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from multiprocessing import cpu_count
import threading
from pathlib import Path
from collections import OrderedDict


from clang.cindex import (
    Index, 
    CursorKind, 
    TypeKind, 
    TokenKind,
    TranslationUnit,
    CompilationDatabase, 
    CompilationDatabaseError,
    Config
)


from utils_api import (
    # normal
    read_json,
    write_json,
    read_file,
    write_file,
    delete_file,
    copy_file,
    create_directory,
    recreate_directory,
    delete_directory,
    copy_directory,
    create_backup_directory,
    restore_directory,
    tmp_backup_directory,
    run_script,
    run_script_wo_log,
    find_compile_commands_json,
    deduplicate_compile_commands,
    get_abs_path,
    append_json,
    get_last_directory,
    get_all_files,
    read_specific_lines,
    get_random,
    read_json_streaming,
    check_permission,

    # translation
    get_name_key,
    obtain_metadata,
    reverse_meta_path,
    parse_def_loc,
)

from llm_api import (
    ask_llm,
    RepairConfig,
    #LLMConfig,
    LLMInterface,
    init_prompt_count, 
    #set_exp_data,
    repair_test,
    repair_branch,
    occupy_llm,
    configure_llm,
    shutdown_llm,
    save_coverage_report
)

# from .translate_core import (
#     obtain_metadata,
#     read_specific_lines,
# )


LLM_ON = False
WEIGHT = None

CXTranslationUnit_None = 0x0
CXTranslationUnit_DetailedPreprocessingRecord = 0x01
CXTranslationUnit_Incomplete = 0x02
CXTranslationUnit_PrecompiledPreamble = 0x04
CXTranslationUnit_CacheCompletionResults = 0x08
CXTranslationUnit_ForSerialization = 0x10
CXTranslationUnit_CXXChainedPCH = 0x20
CXTranslationUnit_SkipFunctionBodies = 0x40
CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = 0x80
CXTranslationUnit_CreatePreambleOnFirstParse = 0x100
CXTranslationUnit_KeepGoing = 0x200
CXTranslationUnit_SingleFileParse = 0x400
CXTranslationUnit_LimitSkipFunctionBodiesToPreamble = 0x800
CXTranslationUnit_IncludeAttributedTypes = 0x1000
CXTranslationUnit_VisitImplicitAttributes = 0x2000
CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles = 0x4000
CXTranslationUnit_RetainExcludedConditionalBlocks = 0x8000




@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def has_include_next(file_path, dep_map, target_dir):
    """
    Returns True if the file source contains #include_next and the file includes files outside target_dir in its dependency chain
    """
    found = False
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.startswith('#') and 'include_next' in stripped:
                found = True
                break
    
    if not found:
        return False
    
    # Check if #include_next actually resolves to a file outside target_dir
    entry = dep_map.get(file_path)
    if not entry:
        return False
    
    for inc in entry.get('include', []):
        inc_path = inc.rsplit(":", 2)[0]
        if target_dir not in inc_path:
            return True
    
    return False


def check_is_system(file_path, target_dir, dep_map):
    """For internal use only. Called only from generate_is_system"""
    if target_dir not in file_path:
        return True
    
    entry = dep_map.get(file_path)
    if not entry:
        return False
    
    included_by = entry.get('included_by', [])
    
    if len(included_by) == 0:
        return False
    
    # added
    if has_include_next(file_path, dep_map, target_dir):
        return True
    # ended
    
    for inc_entry in included_by:
        parts = inc_entry.rsplit(":", 2)
        including_path = parts[0]
        if target_dir in including_path:
            return False
    
    return True


def generate_is_program(target_dir, dep_json_path, is_program_path):
    dep_map = {}
    
    for entry in read_json_streaming(dep_json_path):
        dep_map[entry['source']] = entry
    
    program_files = set()
    for file_path, entry in dep_map.items():
        if str(target_dir) not in str(file_path):
            continue
        
        # Exclude items that are system-like, even within the target_dir
        if check_is_system(file_path, target_dir, dep_map):
            continue
        
        program_files.add(file_path)
    
    write_json(is_program_path, list(program_files))
    return program_files



def is_system_file(use_file, program_files):
    #program_files = set(read_json(is_program_path))

    if use_file not in program_files:
        return True

    return False



def get_build_path(raw_dir):
    """Find and return the path to the c_build.sh file"""
    for root, dirs, files in os.walk(raw_dir):
        if 'c_build.sh' in files:
            return os.path.join(root, 'c_build.sh')
    return None


def get_dep_files(dep_json_path):
    dep_files = set()
    dep_json = read_json(dep_json_path)
    for item in dep_json:
        dep_files.update(item['indirect_include'])

    return list(dep_files)

    

def is_static_function(node) -> bool:
    try:
        tokens = list(node.get_tokens())
        return any(token.spelling == "static" for token in tokens)
    except Exception as e:
        print(f"Error checking static: {e}", file=sys.stderr)
        return False


def get_param_types(node):
    try:
        param_types = []
        for param in node.get_arguments():
            param_types.append(param.type.spelling)
        
        return param_types
    except:
        return [] #"unknown"


def get_actual_type(arg_type, pointer_count):
    if arg_type is None:
        return None
    if pointer_count < 1:
        return arg_type
    return arg_type + " " + ("*" * pointer_count)




def get_var_type(node, libclang):
    #kind = get_kind(node.type)
    array_dimensions = None
    pointer_count = 0
    register_type = None
    is_restrict = False
    is_volatile = False

    if libclang is True:
        if hasattr(node, 'type'):  # If the node is Cursor
            var_type = node.type.spelling
            is_restrict = node.type.is_restrict_qualified()
            is_volatile = node.type.is_volatile_qualified()
        else:  # If the node is Type
            var_type = node.spelling
            is_restrict = node.is_restrict_qualified()
            is_volatile = node.is_volatile_qualified()

    else:
        var_type = node

    #initial_case_type = var_type
    #var_type = var_type


    # if libclang is True: # added # Do we need this?
    #     user_defined_type = get_user_defined_type(node) #.type)
    #     if user_defined_type == "typedef":
    #         base_type = node.spelling
    #         # var_type = base_type

    changed = False
    pointer = False

    if '(' in var_type: # 'FunctionProto', 
        var_type = var_type.split()[0].strip()
        changed = True

    if '[' in var_type: #kind == "CONSTANTARRAY": # #if '[' in var_type:
        changed = True
        pointer = True

        array_dimensions = var_type.count('[')
        pointer_count = array_dimensions

        var_type = var_type.split('[')[0].strip()
    
    if '*' in var_type:
        pointer_count += count_pointers(var_type)
        var_type = var_type.replace('*', '').strip()
        changed = True
        pointer = True

    if var_type.startswith('const '):
        var_type = var_type[6:].strip()
        changed = True
    
    if var_type.startswith('enum '):
        var_type = var_type[5:].strip()
        changed = True
    
    if var_type.startswith('enum'): 
        var_type = "int" #var_type[4:].strip()
        changed = True

    if 'const' in var_type:
        var_type = clean_word(var_type, 'const')
        changed = True
    
    if is_restrict:
        if 'restrict' in var_type:
            var_type = clean_word(var_type, 'restrict')
            changed = True
    
    if is_volatile:
        if 'volatile' in var_type:
            var_type = clean_word(var_type, 'volatile')
            changed = True

    if var_type.startswith('struct '):
        var_type = var_type[7:].strip()
        changed = True
    
    #if changed is True:  # correct
    #var_type = base_type

    #if pointer is True:
    #    var_type = var_type + " ptr"

    var_type = var_type.replace(' ', '_')

    if pointer_count == 1:
        register_type= f"{var_type}_ptr"
    elif pointer_count == 2:
        register_type = f"{var_type}_ptr_ptr"
    elif pointer_count == 3:
        register_type = f"{var_type}_ptr_ptr_ptr"    
    
    
    if register_type is None:
        register_type = var_type

    # if register_type is None:
    #     register_type = var_type

    return var_type, pointer, pointer_count, array_dimensions, register_type


def get_base_type(type_obj):
    canonical_type = type_obj.get_canonical()
    # return {
    #     'base_type': canonical_type.spelling,
    #     'kind': canonical_type.kind.spelling
    # }
    return canonical_type.spelling



def get_arg_type(node, libclang):
    array_dimensions = None
    pointer_count = 0
    register_type = None
    is_restrict = False

    if libclang is True:
        type_name = node.spelling #get_base_type(node.type)
        is_restrict = node.is_restrict_qualified()
    else:
        type_name = node


    changed = False
    pointer = False

    if '(' in type_name: # 'FunctionProto', 
        type_name = type_name.split()[0].strip()
        changed = True

    if '[' in type_name: #kind == "CONSTANTARRAY": # #if '[' in type_name:
        changed = True
        pointer = True

        array_dimensions = type_name.count('[')
        pointer_count = array_dimensions

        type_name = type_name.split('[')[0].strip()


    
    if '*' in type_name:
        pointer_count += count_pointers(type_name)
        type_name = type_name.replace('*', '').strip()
        changed = True
        pointer = True

    if type_name.startswith('const '):
        type_name = type_name[6:].strip()
        changed = True
    
    """
    if type_name.startswith('enum '):
        type_name = type_name[5:].strip()
        changed = True
    
    if type_name.startswith('enum'):
        type_name = "int" #type_name[4:].strip()
        changed = True
    """

    if 'const' in type_name:
        type_name = clean_word(type_name, 'const')
        changed = True
    
    if is_restrict:
        if 'restrict' in type_name:
            type_name = clean_word(type_name, 'restrict')
            changed = True

    """
    if type_name.startswith('struct '):
        type_name = type_name[7:].strip()
        changed = True
    """

    return type_name


def find_actual_definition(cursor):
    if cursor is None:
        return None
        
    if cursor.is_definition():
        return cursor
    
    return None #cursor


def get_containing_function(cursor) -> Optional[str]:
        current = cursor
        while current and current.kind != CursorKind.TRANSLATION_UNIT:
            if current.kind == CursorKind.FUNCTION_DECL:
                return current.spelling
            current = current.semantic_parent
        return None

def is_macro_generated(cursor):
    if cursor.location.file is None:
        return True
        
    try:
        with open(cursor.location.file.name, 'r') as f:
            lines = f.readlines()
            line_num = cursor.location.line - 1  # 0-based
            if line_num >= len(lines):
                return True
            line = lines[line_num]
            
            # If the function name is not present on that line, treat it as a macro expansion
            if cursor.spelling not in line:
                return True
    except:
        return True
        
    return False



def get_function_key(function_name, file_path, start_line):
    return f"{function_name}|{file_path}|{start_line}"


def get_return_type(node) -> str:
    try:
        return_type = node.result_type.spelling
        
        return f"{return_type}"
    except:
        return "unknown"

def count_pointers(type_str):
    return type_str.count('*')


def clean_word(type_str, word):
    parts = type_str.split()
    parts = [p for p in parts if p != f'{word}']
    
    if '*' in type_str:
        # Split by * and remove const from each part
        pointer_parts = type_str.split('*')
        pointer_parts = [part.replace(f'{word}', '').strip() 
                        for part in pointer_parts]
        # Join only the non-empty parts with *
        return '*'.join(part for part in pointer_parts if part).strip()
    
    return ' '.join(parts).strip()



def get_metadata(c_path, meta_dir, path_flag):
    is_abs_os_path = os.path.isabs(c_path)
    if is_abs_os_path:
        c_path = os.path.relpath(c_path)

    root, ext = os.path.splitext(c_path)
    suffix = ext.replace(".", "_") if ext else ""

    meta_path = meta_dir + "/" + root + suffix + ".json"

    if path_flag is False:
        meta_data = read_json(meta_path)
        return meta_data

    elif path_flag is True:
        return meta_path

    else:
        meta_data = read_json(meta_path)
        return meta_data, meta_path


def get_file_path_from_meta_path(meta_path, meta_dir=None):
    path = meta_path.replace(".json", "")
    if meta_dir:
        idx = path.find(meta_dir)
        if idx != -1:
            path = path[idx + len(meta_dir.rstrip("/")) + 1:]

    if path.endswith("_c"):
        return "/" + path[:-2] + ".c"
    elif path.endswith("_h"):
        return "/" + path[:-2] + ".h"
    elif path.endswith("_sh"):
        return "/" + path[:-3] + ".sh"
    else:
        return "/" + path
        

def get_abs_metadata(c_path, meta_dir, path_flag):
    if not os.path.isabs(c_path):
        c_path = os.path.abspath(c_path)

    root, ext = os.path.splitext(c_path)
    suffix = ext.replace(".", "_") if ext else ""

    meta_path = meta_dir + "/" + root + suffix + ".json"

    if path_flag is False:
        meta_data = read_json(meta_path)
        return meta_data

    elif path_flag is True:
        return meta_path

    else:
        meta_data = read_json(meta_path)
        return meta_data, meta_path


def get_is_incomplete(var_type, def_file_path, all_files):
    if def_file_path in all_files:
        return False
    
    if var_type in ['size_t', 'var_list']:
        return False

    return True


def write_definitions(func_json_path, process_type, update_flag, meta_dir):

    data = read_json(func_json_path)

    # Group items by def_file_path
    file_path_groups = {}
    for item in data:
        def_file_path = item['def_file_path']
        
        if def_file_path is None:
            continue
            
        if def_file_path not in file_path_groups:
            file_path_groups[def_file_path] = []
            
        file_path_groups[def_file_path].append(item)

    
    for def_file_path, items in file_path_groups.items():
    #for item in data:
        #def_file_path = item['def_file_path']

        if def_file_path is None:
            continue

        meta_data, meta_path = get_metadata(def_file_path, meta_dir, None)
        if not meta_path.startswith(f'{meta_dir}/workspace'):
            continue

        # print(f"def_file_path: {def_file_path}")
        # print(f"meta_path: {meta_path}")

        if meta_data is None:
            meta_data = []
        # else:
        #     if update_flag:
        #         continue

        #def_end_line = find_function_end(def_file_path, item['def_start_line'])

        # Process all items for this file path
        with open(def_file_path, 'r') as file:
            lines = file.readlines()

        for item in items:
            if process_type == "function":
                item['kind'] = "function"
                def_start_line = item['def_start_line']
                item['def_end_line'] = find_function_end(lines, def_start_line)  #tem['def_end_line'] = def_end_line
                item['event'] = "definition"

                item['line_cov'] = None
                item['branch_cov'] = None

                meta_data.append(item)
                """
                meta_data.append({
                    "name" : item['name'],
                    "kind" : "function",
                    "def_file_path" : item['def_file_path'],
                    "def_start_line" : item['def_start_line'],
                    "def_end_line" : def_end_line,
                    "event" : "definition",
                    "line_cov" : None, 
                    "branch_cov" : None
                })
                """
            else:
                item['kind'] = "data_type"
                #item['def_end_line'] = def_end_line
                item['event'] = "definition"
                meta_data.append(item)

                """
                meta_data.append({
                    "name" : item['name'],
                    "kind" : "data_type",
                    "def_file_path" : item['def_file_path'],
                    "def_start_line" : item['def_start_line'],
                    "def_end_line" : def_end_line,
                    "event" : "definition"
                })
                """

        write_json(meta_path, meta_data)


def is_system_include(path):
    system_paths = ['/usr/include', '/usr/local/include']
    return any(path.startswith(sys_path) for sys_path in system_paths)


def collect_includes(translation_unit, file_path):
    """
    Collects only the actually valid include directives after preprocessing.
    
    Args:
        translation_unit: Clang translation unit
        file_path: Path to the target source file
        includes: List to store include information
    """
    includes = []
    if not translation_unit:
        return includes

    try:
        source_dir = os.path.dirname(os.path.abspath(file_path))
        abs_file_path = os.path.abspath(file_path)
        
        for inc in translation_unit.get_includes():

            if inc.source and inc.location and inc.source.name == file_path:

                include_path = inc.include.name
                if include_path.startswith('./'):
                    source_dir = os.path.dirname(file_path)
                    include_path = include_path[2:]
                    
                    include_path = os.path.normpath(os.path.join(source_dir, include_path))
                
                elif not is_system_include(include_path):
                    #source_dir = os.path.dirname(include_path)
                    #include_path = source_dir + "/" + os.path.relpath(include_path)
                    include_path = os.path.relpath(include_path)
                else:
                    include_path = os.path.abspath(include_path)

                include_path = os.path.abspath(include_path) # added
                include_info = {
                    'source': inc.source.name,
                    'include': include_path, #inc.include.name,
                    'depth': inc.depth,
                    'is_system': is_system_include(include_path),
                    'line': inc.location.line
                }
                includes.append(include_info)
                # print(f"Found include in {file_path}:")
                # print(f"  Include: {include_path}")
                # print(f"  Line: {inc.location.line}")
                # print(f"  Depth: {inc.depth}")

    except Exception as e:
        print(f"Error collecting includes from {file_path}: {str(e)}")

    return includes


def get_files_list(list_path):
    order = []
    with open(list_path, 'r') as file:
        for line in file:
            order.append(line.strip())
    return order


def search_executable(list_path, meta_dir):

    executable = False
    main_list = [] 
    order = get_files_list(list_path)

    print(order)
    for file_path in order:
        meta_data, meta_path = get_metadata(file_path, meta_dir, None)
        if meta_data is None: # added
            continue

        for item in meta_data:
            if item['name'] == 'main':
                executable = True
                main_entry = {}
                main_entry = item.copy()
                main_list.append(main_entry)
                # break
        # if executable:
        #     break

    return executable, main_list


def get_indirect_includeds(file_path, dep_json_path):

    indirect_includes = []
    dep_data = read_json(dep_json_path)
    for item in dep_data:
        if item['source'] == file_path:
            indirect_includes = item['indirect_included']
            break

    return indirect_includes



def build_dependency_graph(data: List[Dict[str, List[str]]]) -> Dict[str, Set[str]]:
    graph = {}
    for item in data:
        source = item['source']
        dependencies = set(item['include'])
        graph[source] = dependencies
    return graph


def find_all_dependencies(graph: Dict[str, Set[str]], start: str, visited: Set[str] = None) -> Set[str]:
    if visited is None:
        visited = set()
    for dependency in graph.get(start, []):
        if dependency not in visited:
            visited.add(dependency)
            find_all_dependencies(graph, dependency, visited)
    return visited



def add_indirect_include(dep_json_path):
    data = read_json(dep_json_path)
    graph = build_dependency_graph(data)

    for item in data:
        source = item['source']
        all_deps = find_all_dependencies(graph, source)
        direct_deps = graph[source]
        indirect_deps = sorted(all_deps - direct_deps)  # all_deps - direct_deps
        direct_include = item['include']
        total_set = set(direct_include).union(set(indirect_deps))
        item['indirect_include'] = list(total_set) #list(indirect_deps)
    
    write_json(dep_json_path, data) # with open(updated_json_path, 'w') as f:

    

def add_included(json_file_path, sign):
    data = read_json(json_file_path)
    
    if sign == "direct":
        key = "included_by" #"included"  
    else:
        key = "indirect_included_by"
    for item in data:
        item[key] = []

    for target in data:
        target_source = target['source']
        for item in data:
            if sign == "direct":
                if target_source in item['include']:
                    target[key].append(item['source'])
            else:
                if target_source in item['include'] or target_source in item['indirect_include']:
                    target[key].append(item['source'])
        
    write_json(json_file_path, data)



def analyze_function(target_dir, meta_dir, dep_json_path, build_dir, database_dir,
                     macro_finder, div_meta_dir, build_path, 
                     taken_directive_path, unordered_taken_directive_path, all_directive_path,
                     all_macros_path, taken_macros_path, guards_path, guarded_macros_path, independent_path, flag_path, const_path, cfg_path,
                     is_program_path
                     ):  # func_json_path, prot_json_path, order_path, False
    
    # 1st round: parsing # If not split into multiple parts, the line numbers will change
    parse_all("call", macro_finder, target_dir, meta_dir, div_meta_dir, database_dir, build_path, 
                 taken_directive_path, unordered_taken_directive_path, all_directive_path, dep_json_path, is_program_path, 
                 all_macros_path, taken_macros_path, guards_path, guarded_macros_path, independent_path, flag_path, const_path,
                 None, None, global_path) #given_compile_dir, given_compile_json_path) # , cfg_path
    


def analyze_dependencies(target_dir, meta_dir, database_dir, dep_json_path):

    compile_dir = find_compile_commands_json(target_dir)
    compile_dir = Path(compile_dir)
    compile_json = compile_dir / "compile_commands.json"
    find_headers(target_dir, database_dir, dep_json_path, compile_dir, compile_json, None)

    # insert "indirect dependencies" items --> create dep_json file
    add_indirect_include(dep_json_path)

    # insert "included" items
    add_included(dep_json_path, "indirect")


def get_metrics(item, graph_metrics):

    target_function_id = f"{item['name']}@{item['def_file_path']}:{item['def_start_line']}"
    
    result = {
        "in_degree": graph_metrics["in_degree"].get(target_function_id, 0),
        "out_degree": graph_metrics["out_degree"].get(target_function_id, 0),
        "degree_centrality": graph_metrics["degree_centrality"].get(target_function_id, 0),
        "betweenness_centrality": graph_metrics["betweenness_centrality"].get(target_function_id, 0),
        "pagerank": graph_metrics["pagerank"].get(target_function_id, 0),
        "eigenvector_centrality": graph_metrics["eigenvector_centrality"].get(target_function_id, 0),
        "closeness_centrality": graph_metrics["closeness_centrality"].get(target_function_id, 0),
        "katz_centrality": graph_metrics["katz_centrality"].get(target_function_id, 0),  
        "combined_score": graph_metrics["combined_score"].get(target_function_id, 0)
    }
    
    return result #graph_metrics["pagerank"].get(target_function_id, 0)


def sort_by_centrality(fixed_metric, data, graph_metrics):
    for ley, item in data.items():
        item["metrics"] = get_metrics(item, graph_metrics)

    sort_by = "closeness_centrality"

    if fixed_metric is not None:
        sort_by = fixed_metric
    
    if sort_by == "pagerank":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["pagerank"], reverse=True)
    elif sort_by == "degree_centrality":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["degree_centrality"], reverse=True)
    elif sort_by == "betweenness_centrality":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["betweenness_centrality"], reverse=True)
    elif sort_by == "closeness_centrality":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["closeness_centrality"], reverse=True)
    elif sort_by == "eigenvector_centrality":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["eigenvector_centrality"], reverse=True)
    elif sort_by == "katz_centrality":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["katz_centrality"], reverse=True)
    elif sort_by == "combined_score":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["combined_score"], reverse=True)
    elif sort_by == "in_degree":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["in_degree"], reverse=True)
    elif sort_by == "out_degree":
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["out_degree"], reverse=True)
    else:
        sorted_keys = sorted(data.keys(), key=lambda k: data[k]["metrics"]["pagerank"], reverse=True)
    
    sorted_data = {k: data[k] for k in sorted_keys}
    
    return sorted_data



katz_alpha = 0.5
katz_beta = 1 #100

def build_graph(callee_path):  #data, callee_main_path):
    data = read_json(callee_path)
    G = nx.DiGraph()
    
    #related_ids = get_related_data(callee_main_path)
 
    for func_id, func_data in data.items():
        # if func_id not in related_ids:
        #     continue
        
        if len(func_data['callers']) == 0 and len(func_data['callees']) == 0: # added
            continue
        G.add_node(func_id, name=func_data["name"])
        

        for callee in func_data.get("callees", []):
            G.add_edge(func_id, callee)
    
    graph_metrics = {}
    
    # 1. Degree-related metrics
    graph_metrics["in_degree"] = dict(G.in_degree())
    graph_metrics["out_degree"] = dict(G.out_degree())
    graph_metrics["degree_centrality"] = nx.degree_centrality(G)
    
    # 1.1 Calculate in-degree centrality and out-degree centrality (added)
    node_count = G.number_of_nodes()
    if node_count <= 1:
        graph_metrics["in_degree_centrality"] = {node: 0.0 for node in G.nodes()}
        graph_metrics["out_degree_centrality"] = {node: 0.0 for node in G.nodes()}
    else:
        # Calculate in-degree centrality
        graph_metrics["in_degree_centrality"] = {
            node: graph_metrics["in_degree"][node] / (node_count - 1) 
            for node in G.nodes()
        }
        
        # Calculate out-degree centrality as well (optional)
        graph_metrics["out_degree_centrality"] = {
            node: graph_metrics["out_degree"][node] / (node_count - 1) 
            for node in G.nodes()
        }
    
    # 2. Betweenness centrality (using sampling)
    MAX_NODE_COUNT = 300 # 100
    # Check the number of nodes in the graph and set the k parameter appropriately
    if node_count == 0:
        # Return an empty dictionary if no nodes exist
        graph_metrics["betweenness_centrality"] = {}

    elif node_count < MAX_NODE_COUNT:
        # Use all nodes when the node count is small
        graph_metrics["betweenness_centrality"] = nx.betweenness_centrality(G)
    else:
        # Use sampling when the node count is large
        graph_metrics["betweenness_centrality"] = nx.betweenness_centrality(G, k=MAX_NODE_COUNT)
    
    # 3. PageRank
    graph_metrics["pagerank"] = nx.pagerank(G)
     
    # 4. Eigenvector centrality
    try:
        graph_metrics["eigenvector_centrality"] = nx.eigenvector_centrality_numpy(G)
    except:
        graph_metrics["eigenvector_centrality"] = {node: 0.0 for node in G.nodes()}
    
    # 5.Closeness centrality
    try:
        graph_metrics["closeness_centrality"] = nx.closeness_centrality(G)
    except:
        graph_metrics["closeness_centrality"] = {node: 0.0 for node in G.nodes()}
    
    # 6. Katz centrality
    try:
        try:
            # First try with the default value
            graph_metrics["katz_centrality"] = nx.katz_centrality(G, alpha=katz_alpha, beta=katz_beta) #beta=1.0)  #(G, alpha=0.1, beta=1.0)
        except nx.PowerIterationFailedConvergence:
            # If it doesn't converge, try a smaller alpha value
            # Calculate the largest eigenvalue of the adjacency matrix to set an appropriate alpha
            A = nx.adjacency_matrix(G).todense()
            try:
                eigenvalues = np.linalg.eigvals(A)
                max_eigenvalue = max(abs(eigenvalues))
                # Set alpha to be smaller than the reciprocal of the largest eigenvalue
                safe_alpha = 0.9 / max_eigenvalue if max_eigenvalue > 0 else 0.01
                graph_metrics["katz_centrality"] = nx.katz_centrality(G, alpha=safe_alpha, beta=1.0)
            except:
                # If eigenvalue computation fails, try a very small alpha value
                graph_metrics["katz_centrality"] = nx.katz_centrality(G, alpha=0.01, beta=1.0)
    except:
        # If it still fails, use default values
        graph_metrics["katz_centrality"] = {node: 0.0 for node in G.nodes()}
    
    # 7. Combined score calculation (including in-degree centrality)
    combined_score = {}
    for node in G.nodes():
        score = (
            graph_metrics["in_degree"].get(node, 0) * 1.0 +
            graph_metrics["out_degree"].get(node, 0) * 0.5 +
            graph_metrics["in_degree_centrality"].get(node, 0) * 2.0 + 
            graph_metrics["betweenness_centrality"].get(node, 0) * 10 +
            graph_metrics["pagerank"].get(node, 0) * 5 +
            graph_metrics["eigenvector_centrality"].get(node, 0) * 3 +
            graph_metrics["closeness_centrality"].get(node, 0) * 2 +
            graph_metrics["katz_centrality"].get(node, 0) * 4
        )
        combined_score[node] = score
    
    graph_metrics["combined_score"] = combined_score
    
    return graph_metrics, G



def build_relationship(function_metadata):
    """
    Build a call graph with caller and callee information from function metadata.
    Uses a composite key of function name, file path, and start line to handle
    functions with the same name.
    
    Args:
        function_metadata: List of JSON metadata entries
    Returns:
        Dictionary with unique function identifiers as keys and function info as values
    """
    # Step 1: Collect basic function information
    functions = {}
    func_id_to_info = {} 
    name_to_ids = {} 
    
    for item in function_metadata:
        if item.get('kind') == 'function':

            func_name = item['name']
            file_path = item.get('def_file_path', '')
            start_line = item.get('def_start_line', 0)
            
            func_id = f"{func_name}@{file_path}:{start_line}"
            
            # if func_name not in name_to_ids:
            #     name_to_ids[func_name] = []
            # name_to_ids[func_name].append(func_id)
            
            func_info = {
                'id': func_id,
                'name': func_name,
                'signature': item.get('signature', ''),
                'file_path': file_path,
                'is_static': item.get('is_static', False),
                'def_start_line': start_line,
                'def_end_line': item.get('def_end_line'),
                'callers': [],
                'callees': [],
                'call_sites': []   
            }
            
            functions[func_id] = func_info
            #func_id_to_info[func_id] = func_info
    
    # Step 2: Build call relationships
    for item in function_metadata:
        if item.get('kind') == 'function' and item.get('callees'):
            caller_name = item['name']
            caller_file = item.get('def_file_path', '')
            caller_line = item.get('def_start_line', 0)
            caller_id = f"{caller_name}@{caller_file}:{caller_line}"
            
            if caller_id not in functions:
                continue

            for call_site in item.get('callees', []):
                callee_name = call_site['name']
                callee_file = call_site.get('def_file_path', '')
                callee_line = call_site.get('def_start_line', 0)
                
                callee_id = f"{callee_name}@{callee_file}:{callee_line}"


                call_file = call_site.get('call_file_path', '')
                call_line = call_site.get('call_start_line', 0)
                callsite_id = f"{callee_name}@{call_file}:{call_line}"

                if callee_id not in functions:
                    functions[callee_id] = {
                        'id': callee_id,
                        'name': callee_name,
                        'signature': '',
                        'file_path': callee_file,
                        'is_static': False,
                        'def_start_line': callee_line,
                        'def_end_line': None,
                        'callers': [],
                        'callees': [],
                        'call_sites': [],
                        'is_external': True  
                    }
                
                # if callee_name not in name_to_ids:
                #     name_to_ids[callee_name] = []
                # name_to_ids[callee_name].append(callee_id)
                if caller_id not in functions:
                    functions[caller_id] = {}
            
                if callee_id not in functions[caller_id]['callees']:
                    functions[caller_id]['callees'].append(callee_id)
                
                if callee_id not in functions[caller_id]['call_sites']:
                    functions[caller_id]['call_sites'].append(callsite_id)

                if caller_id not in functions[callee_id]['callers']:
                    functions[callee_id]['callers'].append(caller_id)

    return functions



def analyze_call_relationship(meta_dir, callee_path, target_dir, is_program_path):
    """Analyze call relationships from the metadata directory"""
    print("Analyzing call_relationship...")

    meta_files = []
    target_paths = []
    parent_paths = []

    program_files = set(read_json(is_program_path))
    for root, _, files in os.walk(meta_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                #print(file_path)
                def_file_path = get_file_path_from_meta_path(file_path, meta_dir)

                # if target_dir not in def_file_path:
                #     continue
                if is_system_file(def_file_path, program_files):
                    continue

                meta_files.append(file_path)

    # 1. Load metadata
    #print(f"meta_files: {meta_files}")
    function_metadata = []
    for meta_path in meta_files:
        data = read_json(meta_path)
        function_metadata.extend(data)

    if not function_metadata:
        print("Failed to load metadata")
        return
    
    print(f"Loaded function metadata: {len(function_metadata)} entries")
    
    # 2. Build call graph
    call_graph = build_relationship(function_metadata)
    print(f"Built call graph: {len(call_graph)} functions")
    

    write_json(callee_path, call_graph)
    print("Saved call relationships to callee.json")



def get_edge_weight(callee_path, callee_main_path, distance_path):
    functions = read_json(callee_path)

    weight_path = 'weight.json'
    path_data = []
    result = {} #[]
    related_data = get_related_data(callee_main_path)
    for key, func_item in functions.items():
        if key not in related_data:
            continue
        call_site_list = func_item['call_sites']
        total = 0
        count = 0
        for item in func_item['callees']:
            #for i in range(0, len(item['path'])):
            source_file = func_item['file_path']  #"sample4.c"
            func_a_name = func_item['name']  #item['path'][i]  #"analyze_input"
            a_start_line = func_item['def_start_line'] 
            a_end_line = func_item['def_end_line'] 

            # func_b_name = item['path'][i+1] #"log_message"
            # line_number = item['call_lines'][i]  #51

            callee_name, callee_path, callee_line = parse_function_id(item)
            callee_list = get_call_site(callee_name, call_site_list)
            #print(callee_list)
            #name, file_path, start_line

            for callee_item in callee_list:
                callee_file_path = callee_item['callee_file_path']
                call_line = callee_item['call_line']

                # print("=======")
                # print(callee_file_path)
                # print(func_a_name)
                # print(callee_name)
                # print(call_line)
                # print("=======")

                if WEIGHT is not None:
                    weight = analyze_branches_to_call(callee_file_path, func_a_name, a_start_line, a_end_line, callee_name, call_line)      
                
                else:
                    weight = 1

                # if 'weight' not in item:
                #     item['weight'] = []
                # item['weight'].append(weight)
                total += weight
                count += 1

            edge_weight = total / count if count != 0 else 0

            pair_id = f"{key}@@{item}"
            result[pair_id]= {}
            result[pair_id] = {
                "source" : key,
                "destination" : item,
                "edge_weight" : edge_weight,
                "path_count" : count
            }
        #print(func_item)
        #print(total)
        #print(count)
        func_item['edge_weight'] = total / count if count != 0 else 0
        func_item['path_count'] = count

    #write_json(weight_path, path_data)
    #write_json(callee_path, functions)

    write_json(distance_path, result)



def get_total_weight(callee_main_path, distance_path):
    functions = read_json(callee_main_path)
    distance = read_json(distance_path)
    
    total_weight = 0
    total_paths = 0

    for function in functions:
        if 'all_paths' in function: # and 'edge_weight' in function:
            #paths_count = len(function['all_paths'])  #total_weight += function['edge_weight']
            
            for i in range(0, len(function['all_paths'])):
                path_list = function['all_paths'][i]['path']
                path_weight = get_path_weight(path_list, distance)
                total_weight += path_weight

        function['total_edge_weight'] = total_weight
        #distance[pair_id]['total_edge_weight']
    
    print(f"\nTotal weight of all {total_paths} paths: {total_weight}")

    write_json(callee_main_path, functions)
    #write_json(distance_path, distance)
    return total_weight



def get_weighted_centrality(callee_path, callee_main_path, distance_path):
    """
    Build a graph for calculating weighted centrality metrics of functions
    
    Args:
        callee_main_path: JSON (dict) containing function relationship information
        distance_path: JSON (list) containing edge weight information
    Returns:
        tuple: (G, metrics_dict) - A tuple containing the graph and a dictionary of centrality metrics
    """
    
    callee_data = read_json(callee_path)
    distance_data = read_json(distance_path)
    related_ids = get_related_data(callee_main_path)

    G = nx.DiGraph()
    
    # 1. Add nodes (from function information)
    for func_id, func_info in callee_data.items():
        if func_id not in related_ids:
            continue
        G.add_node(
            func_id, 
            name=func_info.get('name', ''),
            file_path=func_info.get('file_path', ''),
            start_line=func_info.get('def_start_line', 0)
        )
    
    # 2. Add edges (get weight information from distance_path)
    edge_weights = {}
    for func_key, edge_info in distance_data.items():
        source = edge_info['source']
        dest = edge_info['destination']
        weight = edge_info['edge_weight']
        
        edge_weights[(source, dest)] = weight
        
        if source in G and dest in G:
            G.add_edge(source, dest, weight=weight)
    
    # 3. Add edges from function call relationships (default weight 1.0 if no weight found)
    for func_id, func_info in callee_data.items():
        if 'callees' in func_info:
            for callee in func_info['callees']:
                if (func_id, callee) not in edge_weights:
                    G.add_edge(func_id, callee, weight=1.0)
    
    # 4. Calculate weighted centrality metrics
    metrics = {
        'in_degree': {},
        'out_degree': {},
        'closeness_centrality': {},
        'betweenness_centrality': {},
        'pagerank': {},
        'katz_centrality': {}
    }
    
    # a. Weighted degree centrality (in and out)
    for node in G.nodes():
        # In-degree centrality (number of times called and weight)
        in_edges = G.in_edges(node, data=True)
        metrics['in_degree'][node] = sum(edge[2].get('weight', 1.0) for edge in in_edges)
        
        # Out-degree centrality (number of calls made and weight)
        out_edges = G.out_edges(node, data=True)
        metrics['out_degree'][node] = sum(edge[2].get('weight', 1.0) for edge in out_edges)
    
    # Normalization (optional)
    max_in = max(metrics['in_degree'].values()) if metrics['in_degree'] else 1.0
    max_out = max(metrics['out_degree'].values()) if metrics['out_degree'] else 1.0
    
    for node in G.nodes():
        metrics['in_degree'][node] /= max_in
        metrics['out_degree'][node] /= max_out
    
    # b. Weighted closeness centrality
    try:
        metrics['closeness_centrality'] = nx.closeness_centrality(
            G, distance='weight', wf_improved=True
        )
    except:
        # Calculate individually for disconnected graphs
        for node in G.nodes():
            try:
                path_lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
                if len(path_lengths) > 1:  # Exclude isolated nodes
                    metrics['closeness_centrality'][node] = (len(path_lengths) - 1) / sum(path_lengths.values())
                else:
                    metrics['closeness_centrality'][node] = 0.0
            except:
                metrics['closeness_centrality'][node] = 0.0
    
    # c. Weighted betweenness centrality
    try:
        metrics['betweenness_centrality'] = nx.betweenness_centrality(
            G, weight='weight', normalized=True
        )
    except:
        metrics['betweenness_centrality'] = {node: 0.0 for node in G.nodes()}
    
    # d. Weighted PageRank
    try:
        metrics['pagerank'] = nx.pagerank(
            G, alpha=0.85, weight='weight'
        )
    except:
        metrics['pagerank'] = {node: 0.0 for node in G.nodes()}
    
    # e.
    try:
        A = nx.adjacency_matrix(G, weight='weight').todense()
        
        try:
            eigvals = np.linalg.eigvals(A)
            max_eigval = max(abs(val) for val in eigvals)
            alpha = 0.85 / max_eigval if max_eigval > 0 else 0.1
        except:
            # Use default value if eigenvalue computation fails
            alpha = 0.1
        
        metrics['katz_centrality'] = nx.katz_centrality(
            G, alpha=alpha, beta=1.0, weight='weight', normalized=True
        )
    except Exception as e:
        print(f"Error occurred in Katz centrality calculation: {e}")
        metrics['katz_centrality'] = {node: 0.0 for node in G.nodes()}
    
    return G, metrics



def write_weighted_centrality(callee_path, callee_main_path, distance_path):

    G, metrics = get_weighted_centrality(callee_path, callee_main_path, distance_path)

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    functions = read_json(callee_main_path)
    for item in functions:
        func_id = item['function_id']  #"example_function@example.py"
        node_metrics = get_node_metric(G, metrics, func_id)
        item['metrics'] = node_metrics
    
    write_json(callee_main_path, functions)


def get_all_related_functions(callee_path, function_id, callee_main_path):
    """
    Using graph theory, find all functions reachable from a specific function, including shortest path information to each
    
    Args:
        function_data (dict): Data containing dependencies between functions
        function_id (str): ID of the function to analyze
        callee_main_path (str): Path to the JSON file to write results to
    Returns:
        dict: Dictionary containing information about related functions
    """
    function_data = read_json(callee_path)
    if function_id not in function_data:
        print(f"Function not found: {function_id}")
        return {"callers": [], "callees": [], "all_related": []}
    
    forward_graph, backward_graph = _build_graph(function_data) #, callee_main_path)
    
    result = {
        "callers": [],
        "callees": [],
        "all_related": []
    }
    
    # Find shortest paths using BFS (forward direction - callees)
    def _find_shortest_paths_forward(start_node):
        visited = {start_node: [start_node]}
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            current_path = visited[node]
            
            for neighbor in forward_graph.get(node, []):
                if neighbor not in visited:
                    new_path = current_path + [neighbor]
                    visited[neighbor] = new_path
                    queue.append(neighbor)
        
        return visited
    
    # Find shortest paths using BFS (backward direction - callers)
    def _find_shortest_paths_backward(start_node):
        visited = {start_node: [start_node]}
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            current_path = visited[node]
            
            for neighbor in backward_graph.get(node, []):
                if neighbor not in visited:
                    new_path = current_path + [neighbor]
                    visited[neighbor] = new_path
                    queue.append(neighbor)
        
        return visited
    
    forward_paths = _find_shortest_paths_forward(function_id)
    backward_paths = _find_shortest_paths_backward(function_id)
    
    # Get direct callees from forward_paths
    for callee_id in forward_graph.get(function_id, []):
        if callee_id == function_id:
            continue
            
        details = _get_function_details(callee_id, function_data)
        details["relationship"] = "calls"
        
        path = forward_paths.get(callee_id, [function_id, callee_id])
        details["path"] = [function_data[p].get("name", p) if p in function_data else p for p in path]
        
        result["callees"].append(details)
    
    # Get direct callers from backward_paths
    for caller_id in backward_graph.get(function_id, []):
        if caller_id == function_id:
            continue
            
        details = _get_function_details(caller_id, function_data)
        details["relationship"] = "called_by"
        
        path = backward_paths.get(caller_id, [function_id, caller_id])
        details["path"] = [function_data[p].get("name", p) if p in function_data else p for p in path]
        
        result["callers"].append(details)
    
    self_info = _get_function_details(function_id, function_data)
    self_info["relationship"] = "self"
    self_info["path"] = [function_data[function_id].get("name", function_id)]
    result["all_related"].append(self_info)
    
    # All reachable functions (including direct callees/callers)
    processed = {function_id}  # To avoid duplicates
    
    # Process all reachable callees
    for func_id, path in forward_paths.items():
        if func_id in processed or func_id == function_id:
            continue
            
        processed.add(func_id)
        details = _get_function_details(func_id, function_data)
        
        if func_id in backward_paths:
            details["relationship"] = "bidirectional"
        else:
            details["relationship"] = "forward_reachable"
        
        details["path"] = [function_data[p].get("name", p) if p in function_data else p for p in path]
        
        result["all_related"].append(details)
    
    # Process all reachable callers
    for func_id, path in backward_paths.items():
        if func_id in processed or func_id == function_id:
            continue
            
        processed.add(func_id)
        details = _get_function_details(func_id, function_data)
        
        # Set relationship type (skip those already processed in forward pass)
        details["relationship"] = "backward_reachable"
        
        details["path"] = [function_data[p].get("name", p) if p in function_data else p for p in path]
        
        result["all_related"].append(details)
    
    output_data = {
        "function_id": function_id,
        "name": function_data[function_id].get("name", function_id),
        "total_related": len(result["all_related"]),
        "direct_callers": len(result["callers"]),
        "direct_callees": len(result["callees"]),
        "callees": result["callees"],
        "callers": result["callers"],
        "all_related": result["all_related"]
    }
    
    write_json(f"{database_dir}/related_sum.json", result)

    #os.makedirs(os.path.dirname(callee_main_path), exist_ok=True)
    with open(callee_main_path, 'w', encoding='utf-8') as f:
        json.dump(output_data['all_related'], f, indent=4, ensure_ascii=False)
    
    return result



def get_main_info(main_path, meta_dir):

    main_path = f"{current_path}/{work_dir}/" + main_path
    meta_data, meta_path = get_metadata(main_path, meta_dir, None)

    #print(meta_path)
    for item in meta_data:
        if item['name'] == "main":
            line_number = item['def_start_line']

    target_entry = {
        "target_path" : main_path,
        "target_line" : line_number,
        "target_function" : "main"
    }

    return target_entry



def get_related_main(main_path, meta_dir, callee_main_path, callee_path, distance_path):
    # target_path = target_entry['target_path']
    # target_line = target_entry['target_line']
    # target_function_name = target_entry['target_function']

    target_entry = get_main_info(main_path, meta_dir)

    start_line, end_line = get_bound(target_path, target_function_name, target_line)
    target = {
        "file_path" : target_path,
        "start_line" : start_line,
        "end_line" : end_line,
        "name" : target_function_name
    }

    print(target)

    # call_relations = {}
    # call_relations, functions = get_call_relations(target_entry, meta_dir, visited=None)
    # write_json(callee_main_path, functions)

    functions = []
    target_key = f"{target_function_name}@{target_path}:{start_line}"
    get_all_related_functions(callee_path, target_key, callee_main_path)
    print(callee_main_path)

    call_graph = read_json(callee_path)
    functions = read_json(callee_main_path)

    # # Add main function
    # functions.append({
    #     "function_id": target_key,
    #     "file_path": target_path,
    #     "start_line": start_line,
    #     "end_line": end_line,
    #     "name": target_function_name,
    #     "relationship": None,
    #     "path": [],
    #     "dep_degree": None
    # })

    # Calculate degree
    for call in functions:
        if not (target['file_path'] == call['file_path'] and 
                target['start_line'] == call['start_line'] and 
                target['name'] == call['name']):
            
            func_a = f"{target['name']}@{target['file_path']}:{target['start_line']}"
            func_b = f"{call['name']}@{call['file_path']}:{call['start_line']}"

            steps, path, directions, rel_type = find_shortest_path(call_graph, func_a, func_b)
            all_paths, call_site_list = find_all_paths(call_graph, func_a, func_b)
            call['all_paths'] = all_paths
            call['call_site_list'] = call_graph[func_a]['call_sites'] #get_call_site_list(func_b, call_graph)

            # if call['name'] == "setup_overhead":
            #     print("------")
            #     print(steps)
            #     print(path)
            #     print(func_a)
            #     print(func_b)
            #     print(all_paths)
            #     print("------")

            if steps == -1:
                print("No relationship found between the functions")
            else:
                print(f"Relationship type: {rel_type}")
                print(f"Steps required: {steps}")
                print("Path:")
                for i in range(len(path)-1):
                    print(f"  {path[i]} {directions[i]} {path[i+1]}")
                    print(f"Path: {' -> '.join(path)}")
            
                    
            #call['dep_degree'] = steps
            #call['dep_degree'] = get_dep_degreeget_total_weight((call, target, call_relations)
        else:
            # main itself
            #call['dep_degree'] = 0
            print("Main function itsef")
    
    write_json(callee_main_path, functions)

    get_edge_weight(callee_path, callee_main_path, distance_path)   #get_edge_weight(callee_main_path)

    get_total_weight(callee_main_path, distance_path)

    write_weighted_centrality(callee_path, callee_main_path, distance_path)

    return callee_main_path





#############################################
##### Translation
#############################################


def p_f(process_function, dir, c_flag, h_flag, *args):
    dir = os.path.abspath(dir)

    files_to_process = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            #if (c_flag and file.endswith(".c")) or (h_flag and file.endswith(".h")):
            full_path = os.path.join(root, file)
            files_to_process.append(full_path)

    with ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(process_function, fp, dir, *args): fp
            for fp in files_to_process
        }
        for fut in futures:
            fut.result()

# If process_function writes to the same file (e.g., appending results to a shared JSON), conflicts may occur. Writing to separate output files for each input file avoids this issue.


def replace_comments_with_spaces_file(file_path, raw_dir):

    if not (file_path.endswith(".c") or file_path.endswith(".h")):
        return 

    index = Index.create()
    tu = index.parse(file_path)
    
    #print(file_path)
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            # If the file was read with a non-UTF-8 encoding, convert and save it as UTF-8
            if encoding != 'utf-8':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Converted {file_path} from {encoding} to UTF-8")
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue

    # with open(file_path, 'r') as f:
    #     content = f.readlines()
    
    tokens = tu.get_tokens(extent=tu.cursor.extent)
    comments = []
    
    for token in tokens:
        if token.kind == TokenKind.COMMENT:
            start_line = token.extent.start.line - 1
            start_col = token.extent.start.column - 1
            end_line = token.extent.end.line - 1
            end_col = token.extent.end.column - 1
            
            comment_text = token.spelling
            comment_lines = comment_text.split('\n')
            
            replacements = []
            for i, line in enumerate(comment_lines):
                if len(comment_lines) == 1:
                    # single comment
                    placeholder = ' ' * len(line) 
                    col_start = start_col
                    col_end = end_col
                else:
                    # multiple comments
                    if i == 0:
                        # first line
                        orig_length = len(line)
                        placeholder = ' ' * orig_length
                        col_start = start_col
                        col_end = len(content[start_line + i].rstrip('\n'))
                    elif i == len(comment_lines) - 1:
                        # last line
                        orig_length = end_col
                        placeholder = ' ' * orig_length
                        col_start = 0
                        col_end = end_col
                    else:
                        # intermediate line
                        orig_line = content[start_line + i].rstrip('\n')
                        placeholder = ' ' * len(orig_line)
                        col_start = 0
                        col_end = len(orig_line)
                
                replacements.append((start_line + i, col_start, col_end, placeholder))
            
            comments.extend(replacements)
    
    # Replace comments with whitespace (processing from end to start)
    comments.sort(reverse=True)
    for line_num, start_col, end_col, placeholder in comments:
        if line_num >= len(content):
            continue
        line = content[line_num]
        content[line_num] = line[:start_col] + placeholder + line[end_col:]
    
    result = ''.join(content)
    with open(file_path, 'w') as f:
        f.write(result)
    print(f"Processed file saved as: {file_path}")
    
    return ''.join(content)



def find_c_files_from_compile_db(target_dir):

    target_dir = Path(target_dir)

    compile_db_files = list(target_dir.glob("**/compile_commands.json"))
    
    if not compile_db_files:
        raise FileNotFoundError(
            f"compile_commands.json not found in {target_dir} or subdirectories"
        )
    
    compile_db_path = compile_db_files[0]
    #compile_db_path = target_dir / "compile_commands.json"
    
    if not compile_db_path.exists():
        print(f"ERROR: {compile_db_path} not found")
        return []
    
    print(f"Reading: {compile_db_path}")
    
    with open(compile_db_path, 'r') as f:
        compile_commands = json.load(f)
    
    c_files = []
    for entry in compile_commands:
        file_path = Path(entry['file'])
        if file_path.suffix == '.c':
            c_files.append(file_path)
    
    return c_files



def run_macro_finder(macro_finder, c_file, compile_db_dir, target_dir, output_handle):
    """Run macro-finder and write the results to a file"""
    # print(f"Processing: {c_file}")
    # Add debug information
    target_dir = Path(target_dir)
    
    # # Recursively search for compile_commands.json
    # compile_db_files = list(target_dir.glob("**/compile_commands.json"))
    
    # if not compile_db_files:
    #     raise FileNotFoundError(
    #         f"compile_commands.json not found in {target_dir} or subdirectories"
    #     )
    
    # # If multiple are found, use the first one (or process all)
    # compile_db = compile_db_files[0]

    # compile_db_dir = compile_db.parent

    # #compile_db = target_dir / "compile_commands.json"
    # print(f"  compile_commands.json path: {compile_db}")
    # print(f"  compile_commands.json directory: {compile_db_dir}")
    # print(f"  Exists: {compile_db.exists()}")


    # Build command line arguments
    #cmd = [str(macro_finder), str(c_file), "-p", str(target_dir)]
    cmd = [str(macro_finder), str(c_file), "-p", str(compile_db_dir)]

    # Display command
    cmd_str = ' '.join(cmd)
    #print(f"  Command: {cmd_str}")

    
    output_handle.write(f"\n{'='*80}\n")
    output_handle.write(f"Processing: {c_file}\n")
    output_handle.write(f"{'='*80}\n")
    
    try:
        #print(f"    [RUN] subprocess starting...", flush=True)
        # added
        # ★ Redirect stderr to file to avoid deadlock
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.stdout', delete=True) as stdout_file, tempfile.NamedTemporaryFile(mode='w+', suffix='.stderr', delete=True) as stderr_file:
            result = subprocess.run(
                cmd,
                stdout=stdout_file,   # ★ stdout also goes to file
                stderr=stderr_file,  # ★ stderr goes to file
                text=True,
                timeout=120  # ★ Extended for large files
            )

            #print(f"    [DONE] returncode={result.returncode}", flush=True)
            
            # Read stdout from file and write
            stdout_file.seek(0)
            for line in stdout_file:
                output_handle.write(line)
            
            # Read stderr from file
            stderr_file.seek(0)
            stderr_content = stderr_file.read()
            if stderr_content:
                output_handle.write("\n--- STDERR ---\n")
                output_handle.write(stderr_content)

            output_handle.flush()

            if result.returncode != 0:
                error_msg = f"ERROR: Process returned non-zero exit code: {result.returncode}\n"
                print(f"  ❌ {error_msg}")
                print(f"  >> Run manually: {cmd_str}")  # Display command for manual execution
                output_handle.write(error_msg)
                return False
            
        return True
        
    except subprocess.TimeoutExpired:
        error_msg = f"TIMEOUT: Process exceeded 30 seconds\n"
        print(error_msg)
        output_handle.write(error_msg)
        return False
    
    except Exception as e:
        error_msg = f"CRASHED: {str(e)}\n"
        print(error_msg)
        output_handle.write(error_msg)
        return False


def convert_to_absolute_paths(compile_commands_path='compile_commands.json'):
    """
    Convert relative paths in compile_commands.json to absolute paths
    
    Args:
        compile_commands_path: Path to the compile_commands.json file
    """
    if not os.path.exists(compile_commands_path):
        print(f"Error: {compile_commands_path} not found", file=sys.stderr)
        return False
    
    try:
        with open(compile_commands_path, 'r') as f:
            commands = json.load(f)
        
        for cmd in commands:
            directory = cmd.get('directory', '')
            
            # Convert file path to absolute path
            if 'file' in cmd and not os.path.isabs(cmd['file']):
                cmd['file'] = os.path.abspath(os.path.join(directory, cmd['file']))
            
            # Convert output path to absolute path
            if 'output' in cmd and not os.path.isabs(cmd['output']):
                cmd['output'] = os.path.abspath(os.path.join(directory, cmd['output']))
            
            # Convert paths within arguments
            if 'arguments' in cmd:
                new_args = []
                for arg in cmd['arguments']:
                    # Handle -I option
                    if arg.startswith('-I'):
                        include_path = arg[2:]  # Remove -I
                        if include_path and not os.path.isabs(include_path):
                            abs_path = os.path.abspath(os.path.join(directory, include_path))
                            new_args.append(f'-I{abs_path}')
                        else:
                            new_args.append(arg)
                    # Source files and other file paths
                    elif not arg.startswith('-') and not os.path.isabs(arg):
                        potential_path = os.path.join(directory, arg)
                        if os.path.exists(potential_path):
                            new_args.append(os.path.abspath(potential_path))
                        else:
                            new_args.append(arg)
                    else:
                        new_args.append(arg)
                
                cmd['arguments'] = new_args
        
        # ★ Create backup
        #backup_path = compile_commands_path + '.backup'
        backup_path = str(compile_commands_path) + '.backup'
        write_json(backup_path, commands)

        print(f"Backup created: {backup_path}")
        
        write_json(compile_commands_path, commands)

        #print(f"✅ Successfully converted paths in {compile_commands_path}")
        return True
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False



def run_finder(macro_finder, target_dir, output_file, compile_json_dir, compile_json, round_id):
    # c_files = find_c_files(target_dir)

    target_dir = Path(target_dir)

    # c_files = find_c_files_from_compile_json(target_dir)
    # print(f"Found {len(c_files)} C files")

    """
    # Recursively search for compile_commands.json
    compile_json_files = list(target_dir.glob("**/compile_commands.json"))
    
    if not compile_json_files:
        raise FileNotFoundError(
            f"compile_commands.json not found in {target_dir} or subdirectories"
        )
    # If multiple are found, use the first one (or process all)
    compile_json = compile_json_files[0]
    compile_json_dir = compile_json.parent

    #compile_json = target_dir / "compile_commands.json"
    print(f"  compile_commands.json path: {compile_json}")
    print(f"  compile_commands.json directory: {compile_json_dir}")
    print(f"  Exists: {compile_json.exists()}")
    """

    # with open(compile_json, 'r') as f:
    #     compile_commands = json.load(f)
    compile_commands = read_json(compile_json)

    c_files = []
    for entry in compile_commands:
        file_path = Path(entry['file'])
        # Include both .c and .cpp
        #if file_path.suffix in ['.c', '.cpp', '.cc', '.cxx']:
        c_files.append(file_path)
    
    # ★ Create backup
    random_id = get_random(16)
    """
    create_directory(f"/tmp/{random_id}")
    backup_path = f"/tmp/{random_id}/compile_commands.json.backup"  # This needs to be unique
    copy_file(compile_json, backup_path)

    convert_to_absolute_paths(compile_json)  # This is necessary
    """

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Macro Finder Results\n")
            f.write(f"Target Directory: {target_dir}\n")
            f.write(f"Total Files: {len(c_files)}\n")
            f.write(f"{'='*80}\n\n")
            
            success_count = 0
            fail_count = 0

            """
            for c_file in c_files:
                #print(f"  [START] {c_file}", flush=True)
                if run_macro_finder(macro_finder, c_file, compile_json_dir, target_dir, f):
                    success_count += 1
                    #print(f"  [OK] {c_file}", flush=True)
                else:
                    fail_count += 1
                    #print(f"CRASHED on: {c_file}", flush=True)
                    print(f"CRASHED on: {c_file}")
                    # Break here if you want to stop on error
                    # break
            """
            cmd = [macro_finder, "-p", str(compile_json_dir)]
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            # ★ Always output stderr (includes info like Processing N files... etc.)
            if result.stderr:
                print(f"  stderr:\n{result.stderr}")

            if result.returncode == 0:
                f.write(result.stdout)
                success_count = len(c_files)
            else:
                fail_count = len(c_files)
                print(f"FAILED (exit code: {result.returncode})")
                f.write(f"ERROR (exit code: {result.returncode}):\n{result.stderr}\n")
                # ★ Write stdout too as there may be partial results
                if result.stdout:
                    f.write(result.stdout)

            """
            if result.returncode == 0:
                f.write(result.stdout)
                success_count = len(c_files)
            else:
                fail_count = len(c_files)
                print(f"FAILED: {result.stderr[:500]}")
                f.write(f"ERROR: {result.stderr}\n")
            """
                
                        
            # Output summary
            summary = f"\n{'='*80}\n"
            summary += f"SUMMARY in run_finder (batch mode) @round {round_id}\n"
            summary += f"{'='*80}\n"
            summary += f"Total files processed: {len(c_files)}\n"
            summary += f"Successful: {success_count}\n"
            summary += f"Failed: {fail_count}\n"
            
            print(summary)
            f.write(summary)

            if int(fail_count) > 0:
                print(f"run_finder FAILED. stderr: {result.stderr}", flush=True)
                print(f"run_finder FAILED. returncode: {result.returncode}", flush=True)
                raise ValueError("run_finder failed.")
        
    finally:
        """
        delete_file(compile_json)
        copy_file(backup_path, compile_json)
        delete_directory(f"/tmp/{random_id}")
        """
        print(f"Restored original compile_commands.json")

    print(f"\nResults written to: {output_file}")



# Can it detect both skipped and not skipped macros outside directives?
def run_finder_all(macro_finder, target_dir, output_file):
    # c_files = find_c_files(target_dir)

    target_dir = Path(target_dir)

    # c_files = find_c_files_from_compile_db(target_dir)
    # print(f"Found {len(c_files)} C files")

    
    # Recursively search for compile_commands.json
    compile_db_files = list(target_dir.glob("**/compile_commands.json"))
    
    if not compile_db_files:
        raise FileNotFoundError(
            f"compile_commands.json not found in {target_dir} or subdirectories"
        )
    
    # If multiple are found, use the first one (or process all)
    compile_db = compile_db_files[0]
    compile_db_dir = compile_db.parent

    #compile_db = target_dir / "compile_commands.json"
    print(f"  compile_commands.json path: {compile_db}")
    print(f"  compile_commands.json directory: {compile_db_dir}")
    print(f"  Exists: {compile_db.exists()}")

    with open(compile_db, 'r') as f:
        compile_commands = json.load(f)
    
    c_files = []
    for entry in compile_commands:
        file_path = Path(entry['file'])
        # include both .c and .cpp
        #if file_path.suffix in ['.c', '.cpp', '.cc', '.cxx']:
        c_files.append(file_path)

    # Create backup
    random_id = get_random(16)
    create_directory(f"/tmp/{random_id}")
    backup_path = f"/tmp/{random_id}/compile_commands.json.backup"  # This needs to be unique
    copy_file(compile_db, backup_path)

    convert_to_absolute_paths(compile_db)  # We need this

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Macro Finder Results\n")
            f.write(f"Target Directory: {target_dir}\n")
            f.write(f"Total Files: {len(c_files)}\n")
            f.write(f"{'='*80}\n\n")
            
            success_count = 0
            fail_count = 0
            
            for c_file in c_files:
                if run_macro_finder(macro_finder, c_file, compile_db_dir, target_dir, f):
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"CRASHED on: {c_file}")
                    # Break here if you want to stop on error
                    # break
            
            # Output summary
            summary = f"\n{'='*80}\n"
            summary += f"SUMMARY in run_finder\n"
            summary += f"{'='*80}\n"
            summary += f"Total files processed: {len(c_files)}\n"
            summary += f"Successful: {success_count}\n"
            summary += f"Failed: {fail_count}\n"
            
            print(summary)
            f.write(summary)

            if int(fail_count) > 0:
                raise ValueError("run_finder failed.")
        
    finally:
        delete_file(compile_db)
        copy_file(backup_path, compile_db)
        delete_directory(f"/tmp/{random_id}")
        print(f"Restored original compile_commands.json")

    print(f"\nResults written to: {output_file}")



def get_endif_info(processed_lines, idx, line):
    # Search upward from the current line for [ENDIF]
    for i in range(idx - 1, -1, -1):
        prev_line = processed_lines[i].strip()
        endif_match = re.match(r'\[ENDIF(?: \(skipped\))?\] (.+?):(\d+):(\d+):(\d+):(\d+)', prev_line)
        if endif_match:
            endif_file, e_start_line, e_start_col, e_end_line, e_end_col = endif_match.groups()
            endif_file = get_abs_path(endif_file)
            is_endif_skipped = "(skipped)" in prev_line
            return endif_file, e_start_line, e_start_col, e_end_line, e_end_col, is_endif_skipped
    
    # If not found
    raise ValueError("Must find the end!!")


def put_in_target_dir(file_path, target_dir):
    if "unknown" in file_path:
        return file_path

    if not os.path.isabs(file_path):
        target_dir = os.path.abspath(target_dir)
        file_path = os.path.join(str(target_dir), file_path)

    return file_path



# skipped = whether the parent block was skipped
# evaluated = whether the block was executed: for the delete_untaken_paths function
def save_all_directives(input_file, unordered_macros_path, macros_path, database_dir, target_dir, skipped_flag, evaluated_flag):
    """Parse macro-finder output and save to JSON (complete version based on Close information)"""
    
    data = {
        "files": {},
        "macros": {}
    }
    
    search_key = "appearances" # uses

    current_file = None
    endif_mapping = {}
    current_endif_info = None
    pending_endif = None
    
    seen_entries = {
        "defined": set(),
        "ifdef": set(),
        "ifndef": set(),
        "if": set(),
        "elif": set(),
        "else" : set(),
        "endif": set()
    }
    
    if not os.path.exists(input_file):
        return

    # ★ Step 1: Read the entire file and join lines
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    with open(input_file, 'r', encoding='utf-8') as f:
        initial_lines = f.readlines()

    # ★ Step 2: Completely join continuation lines
    processed_lines = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].rstrip('\n').rstrip('\r')
        
        # ★ For directive lines like [IF], [ELIF], etc.
        #if re.match(r'^\[(IF|ELIF|IFDEF|IFNDEF|ELSE|ENDIF|DEFINED|UNDEFINED)\b', line):
        if re.match(r'^\[(IF|ELIF|IFDEF|IFNDEF|ELSE|ENDIF|DEFINED|UNDEFINED)(?:_\w+)?\]', line):
            while i + 1 < len(raw_lines):
                next_line = raw_lines[i + 1].rstrip('\n').rstrip('\r')
                
                if next_line.lstrip() and next_line.lstrip()[0] == '[':
                    break
                
                if next_line.lstrip().startswith("=> Closes"):
                    break
                
                if next_line.lstrip().startswith("Processing:"):
                    break
                
                if not next_line.strip():
                    i += 1
                    break
                
                line = line.rstrip('\\').rstrip()
                line += ' ' + next_line.lstrip()
                i += 1
        
        # ★ Also join continuation lines for "=> Closes" lines
        elif line.lstrip().startswith("=> Closes"):
            while i + 1 < len(raw_lines):
                next_line = raw_lines[i + 1].rstrip('\n').rstrip('\r')
                
                if next_line.lstrip() and next_line.lstrip()[0] == '[':
                    break
                
                if next_line.lstrip().startswith("=> Closes"):
                    break
                
                if next_line.lstrip().startswith("Processing:"):
                    break
                
                if not next_line.strip():
                    i += 1
                    break
                
                line = line.rstrip('\\').rstrip()
                line += ' ' + next_line.lstrip()
                i += 1
        
        processed_lines.append(line)
        i += 1
    
    # ★ Step 3: Process the joined lines
    for idx, line in enumerate(processed_lines):
        line = line.strip()
        
        if not line:
            continue
        
        if line.startswith("Processing:"):
            current_file = line.split("Processing:")[1].strip()
            continue
        
        # Parse [DEFINED] (active)
        if line.startswith("[DEFINED]") and not line.startswith("[DEFINED (skipped)]"):
            #match = re.match(r'\[DEFINED\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+)', line)
            match = re.match(r'\[DEFINED\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s*->\s*(.+))?$', line)
            if match:
                #file_path, start_line, start_col, end_line, end_col, macro_name = match.groups()
                file_path, start_line, start_col, end_line, end_col, macro_name, definition = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                entry_key = f"DEFINED:{file_path}:{start_line}:{start_col}:{macro_name}:active"
                if entry_key in seen_entries["defined"]:
                    continue
                seen_entries["defined"].add(entry_key)

                macro_key = f"{macro_name}:{file_path}:{start_line}:{start_col}"
                entry = {
                    "kind": "macro",
                    "type": "DEFINED",
                    "name": macro_name,
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "active": True,
                    "skipped": False
                }

                if definition:
                    entry["definition_body"] = definition.strip()
                    
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["defined"].append(entry)
                
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definition": {
                            "file_path": file_path,
                            "start_line": int(start_line),
                            "start_column": int(start_col),
                            "end_line": int(end_line),
                            "end_column": int(end_col),
                            "active": True,
                            "skipped": False,
                            "body": definition.strip() if definition else None
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }

                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definitions": [],
                        "uses": []
                    }
                
                data["macros"][macro_key]["definitions"].append({
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "active": True,
                    "skipped": False
                })
                """
        
        # Parse [DEFINED (skipped)] (inactive)
        elif line.startswith("[DEFINED (skipped)]"):
            if skipped_flag:  # ★ Changed to skipped_flag
                continue

            if evaluated_flag:
                continue  # ★ Added this line

            #match = re.match(r'\[DEFINED \(skipped\)\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+)', line)
            match = re.match(r'\[DEFINED \(skipped\)\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s*->\s*(.+))?$', line)

            if match:
                #file_path, start_line, start_col, end_line, end_col, macro_name = match.groups()
                file_path, start_line, start_col, end_line, end_col, macro_name, definition = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                entry_key = f"DEFINED:{file_path}:{start_line}:{start_col}:{macro_name}:skipped"
                if entry_key in seen_entries["defined"]:
                    continue
                seen_entries["defined"].add(entry_key)

                macro_key = f"{macro_name}:{file_path}:{start_line}:{start_col}"
                entry = {
                    "kind": "macro",
                    "type": "DEFINED",
                    "name": macro_name,
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "active": False,
                    "skipped": True
                }

                if definition:
                    entry["definition_body"] = definition.strip()
                        
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["defined"].append(entry)
                
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definition": {
                            "file_path": file_path,
                            "start_line": int(start_line),
                            "start_column": int(start_col),
                            "end_line": int(end_line),
                            "end_column": int(end_col),
                            "active": False,
                            "skipped": True,
                            "body": definition.strip() if definition else None
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }

                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definitions": [],
                        "uses": []
                    }
                
                data["macros"][macro_key]["definitions"].append({
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "active": False,
                    "skipped": True
                })
                """
        
        # Parse [DEFINED_FUNC] (active)
        elif line.startswith("[DEFINED_FUNC]") and not line.startswith("[DEFINED_FUNC (skipped)]"):
            match = re.match(r'\[DEFINED_FUNC\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s*->\s*(.+))?$', line)
            if match:
                file_path, start_line, start_col, end_line, end_col, macro_sig, definition = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                # Extract macro name (up to the parenthesis)
                macro_name = macro_sig.split('(')[0].strip()
                
                entry_key = f"DEFINED_FUNC:{file_path}:{start_line}:{start_col}:{macro_sig}:active"
                if entry_key in seen_entries["defined"]:
                    continue
                seen_entries["defined"].add(entry_key)

                macro_key = f"{macro_name}:{file_path}:{start_line}:{start_col}"
                entry = {
                    "kind": "macro_function",
                    "type": "DEFINED_FUNC",
                    "name": macro_name,
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "macro_signature": macro_sig,
                    "active": True,
                    "skipped": False
                }
                
                if definition:
                    entry["definition_body"] = definition.strip()
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["defined"].append(entry)
                
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "signature": macro_sig,
                        "definition": {
                            "file_path": file_path,
                            "start_line": int(start_line),
                            "start_column": int(start_col),
                            "end_line": int(end_line),
                            "end_column": int(end_col),
                            "active": True,
                            "skipped": False,
                            "body": definition.strip() if definition else None
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }

                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "signature": macro_sig,
                        "definitions": [],
                        "uses": []
                    }
                
                data["macros"][macro_key]["definitions"].append({
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "active": True,
                    "skipped": False,
                    "body": definition.strip() if definition else None
                })
                """

        # Parse [DEFINED_FUNC (skipped)] (inactive)
        elif line.startswith("[DEFINED_FUNC (skipped)]"):
            if skipped_flag:
                continue

            if evaluated_flag:
                continue

            match = re.match(r'\[DEFINED_FUNC \(skipped\)\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s*->\s*(.+))?$', line)
            if match:
                file_path, start_line, start_col, end_line, end_col, macro_sig, definition = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                macro_name = macro_sig.split('(')[0].strip()
                
                entry_key = f"DEFINED_FUNC:{file_path}:{start_line}:{start_col}:{macro_sig}:skipped"
                if entry_key in seen_entries["defined"]:
                    continue
                seen_entries["defined"].add(entry_key)

                macro_key = f"{macro_name}:{file_path}:{start_line}:{start_col}"
                entry = {
                    "kind": "macro_function",
                    "type": "DEFINED_FUNC",
                    "name": macro_name,
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "macro_signature": macro_sig,
                    "active": False,
                    "skipped": True
                }
                
                if definition:
                    entry["definition_body"] = definition.strip()
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["defined"].append(entry)
                
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "signature": macro_sig,
                        "definition": {
                            "file_path": file_path,
                            "start_line": int(start_line),
                            "start_column": int(start_col),
                            "end_line": int(end_line),
                            "end_column": int(end_col),
                            "active": False,
                            "skipped": True,
                            "body": definition.strip() if definition else None
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }

                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "signature": macro_sig,
                        "definitions": [],
                        "uses": []
                    }
                
                data["macros"][macro_key]["definitions"].append({
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "active": False,
                    "skipped": True,
                    "body": definition.strip() if definition else None
                })
                """
                
        
        # Parse [IFDEF] (active)
        #elif line.startswith("[IFDEF]") and not line.startswith("[IFDEF (skipped)]"):
        elif re.match(r'\[IFDEF(?:_TRUE|_FALSE|_FUNC_TRUE|_FUNC_FALSE)?\]', line) and not line.startswith("[IFDEF (skipped)]"):
            #match = re.match(r'\[IFDEF\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?) \(defined at: (.+?)\)', line)
            match = re.match(r'\[IFDEF(?:_(TRUE|FALSE|FUNC_TRUE|FUNC_FALSE))?\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?) \(defined at: (.+?)\)', line)

            if match:
                eval_value, file_path, start_line, start_col, end_line, end_col, macro_name, def_loc = match.groups()
                evaluated = (eval_value in ["TRUE", "FUNC_TRUE"])
                
                if evaluated_flag and not evaluated:
                    continue

                #file_path, start_line, start_col, end_line, end_col, macro_name, def_loc = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                entry_key = f"IFDEF:{file_path}:{start_line}:{start_col}:{macro_name}:{def_loc}:active"
                if entry_key in seen_entries["ifdef"]:
                    continue
                seen_entries["ifdef"].add(entry_key)

                macro_key = f"{macro_name}:{def_loc}"
                entry = {
                    "kind": "directive",
                    "type": "IFDEF",
                    "name": macro_name,
                    "definition": def_loc, # defined_at
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "active": True,
                    "skipped": False,
                    "evaluated": evaluated
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["ifdef"].append(entry)
                
                use_entry = {
                    "kind": "directive",
                    "type": "IFDEF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "macro_key": macro_key,
                    "active": True,
                    "skipped": False
                }
                
                """
                if macro_key in data["macros"]:
                    data["macros"][macro_key]["uses"].append(use_entry)
                """
                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definitions": [],
                        "uses": []
                    }
                """
                def_file_path, def_line, def_col = parse_def_loc(def_loc)
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definition": {
                            "file_path": def_file_path,
                            "start_line": def_line,
                            "start_column": def_col
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }
                data["macros"][macro_key][search_key].append(use_entry)
                
                # key = f"IFDEF:{file_path}:{start_line}:{start_col}"
                # if key not in endif_mapping:
                #     endif_mapping[key] = {}
                # endif_mapping[key]["use_entry"] = use_entry
                # endif_mapping[key]["file_entry"] = entry
        
        # Parse [IFDEF (skipped)] (inactive)
        elif line.startswith("[IFDEF (skipped)]"):
            if skipped_flag:  # ★ Changed to skipped_flag
                continue

            if evaluated_flag:
                continue  # skipped is always evaluated=False so skip

            match = re.match(r'\[IFDEF \(skipped\)\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?) \(defined at: (.+?)\)', line)
            if match:
                file_path, start_line, start_col, end_line, end_col, macro_name, def_loc = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                entry_key = f"IFDEF:{file_path}:{start_line}:{start_col}:{macro_name}:{def_loc}:skipped"
                if entry_key in seen_entries["ifdef"]:
                    continue
                seen_entries["ifdef"].add(entry_key)

                macro_key = f"{macro_name}:{def_loc}"
                entry = {
                    "kind": "directive",
                    "type": "IFDEF",
                    "name": macro_name,
                    "definition": def_loc, # defined_at
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "active": False,
                    "skipped": True,
                    "evaluated": False
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["ifdef"].append(entry)
                
                use_entry = {
                    "kind": "directive",
                    "type": "IFDEF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "macro_key": macro_key,
                    "active": False,
                    "skipped": True
                }
                """
                if macro_key in data["macros"]:
                    data["macros"][macro_key]["uses"].append(use_entry)
                """
                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definitions": [],
                        "uses": []
                    }
                """
                def_file_path, def_line, def_col = parse_def_loc(def_loc)
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definition": {
                            "file_path": def_file_path,
                            "start_line": def_line,
                            "start_column": def_col
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }
                data["macros"][macro_key][search_key].append(use_entry)
                
                # key = f"IFDEF:{file_path}:{start_line}:{start_col}"
                # if key not in endif_mapping:
                #     endif_mapping[key] = {}
                # endif_mapping[key]["use_entry"] = use_entry
                # endif_mapping[key]["file_entry"] = entry
        
        # Parse [IFNDEF] (active)
        #elif line.startswith("[IFNDEF]") and not line.startswith("[IFNDEF (skipped)]"):
        elif re.match(r'\[IFNDEF(?:_TRUE|_FALSE)?\]', line) and not line.startswith("[IFNDEF (skipped)]"):
            #match = re.match(r'\[IFNDEF\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?) \(defined at: (.+?)\)', line)
            match = re.match(r'\[IFNDEF(?:_(TRUE|FALSE))?\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?) \(defined at: (.+?)\)', line)

            if match:
                eval_value, file_path, start_line, start_col, end_line, end_col, macro_name, def_loc = match.groups()
                evaluated = (eval_value == "TRUE")
                
                if evaluated_flag and not evaluated:
                    continue

                #file_path, start_line, start_col, end_line, end_col, macro_name, def_loc = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)

                entry_key = f"IFNDEF:{file_path}:{start_line}:{start_col}:{macro_name}:{def_loc}:active"
                if entry_key in seen_entries["ifndef"]:
                    continue
                seen_entries["ifndef"].add(entry_key)
                
                macro_key = f"{macro_name}:{def_loc}"

                entry = {
                    "kind": "directive",
                    "type": "IFNDEF",
                    "name": macro_name,
                    "definition": def_loc, # defined_at
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "active": True,
                    "skipped": False,
                    "evaluated": evaluated
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["ifndef"].append(entry)
                
                use_entry = {
                    "kind": "directive",
                    "type": "IFNDEF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "macro_key": macro_key,
                    "active": True,
                    "skipped": False
                }
                """
                if macro_key in data["macros"]:
                    data["macros"][macro_key]["uses"].append(use_entry)
                """
                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definitions": [],
                        "uses": []
                    }
                """
                def_file_path, def_line, def_col = parse_def_loc(def_loc)
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definition": {
                            "file_path": def_file_path,
                            "start_line": def_line,
                            "start_column": def_col
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }
                
                data["macros"][macro_key][search_key].append(use_entry)

                # key = f"IFNDEF:{file_path}:{start_line}:{start_col}"
                # if key not in endif_mapping:
                #     endif_mapping[key] = {}
                # endif_mapping[key]["use_entry"] = use_entry
                # endif_mapping[key]["file_entry"] = entry
        
        # Parse [IFNDEF (skipped)] (inactive)
        elif line.startswith("[IFNDEF (skipped)]"):
            if skipped_flag:  # ★ Changed to skipped_flag
                continue

            if evaluated_flag:
                continue

            match = re.match(r'\[IFNDEF \(skipped\)\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?) \(defined at: (.+?)\)', line)
            if match:
                file_path, start_line, start_col, end_line, end_col, macro_name, def_loc = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                entry_key = f"IFNDEF:{file_path}:{start_line}:{start_col}:{macro_name}:{def_loc}:skipped"
                if entry_key in seen_entries["ifndef"]:
                    continue
                seen_entries["ifndef"].add(entry_key)
                
                macro_key = f"{macro_name}:{def_loc}"

                entry = {
                    "kind": "directive",
                    "type": "IFNDEF",
                    "name": macro_name,
                    "definition": def_loc, # defined_at
                    "macro_key": macro_key,
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "active": False,
                    "skipped": True,
                    "evaluated": False
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["ifndef"].append(entry)
                
                use_entry = {
                    "kind": "directive",
                    "type": "IFNDEF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "macro_key": macro_key,
                    "active": False,
                    "skipped": True
                }
                """
                if macro_key in data["macros"]:
                    data["macros"][macro_key]["uses"].append(use_entry)
                """
                """
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definitions": [],
                        "uses": []
                    }
                """
                def_file_path, def_line, def_col = parse_def_loc(def_loc)
                if macro_key not in data["macros"]:
                    data["macros"][macro_key] = {
                        "name": macro_name,
                        "definition": {
                            "file_path": def_file_path,
                            "start_line": def_line,
                            "start_column": def_col
                        },
                        "is_const" : None,
                        "is_flag" : None,
                        "is_guard" : None,
                        "is_guarded" : None,
                        search_key : []
                    }
                data["macros"][macro_key][search_key].append(use_entry)
                
                # key = f"IFNDEF:{file_path}:{start_line}:{start_col}"
                # if key not in endif_mapping:
                #     endif_mapping[key] = {}
                # endif_mapping[key]["use_entry"] = use_entry
                # endif_mapping[key]["file_entry"] = entry
        
        # Parse [IF] (active)
        #elif line.startswith("[IF]") and not line.startswith("[IF (skipped)]"):
        elif re.match(r'\[IF(?:_TRUE|_FALSE)?\]', line) and not line.startswith("[IF (skipped)]"):
            match = re.match(r'\[IF\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s+\[(.+?)\]\s*)?$', line)
            match = re.match(r'\[IF(?:_(TRUE|FALSE))?\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s+\[(.+?)\]\s*)?$', line)
            if match:
                eval_value, file_path, start_line, start_col, end_line, end_col, condition, macros_str = match.groups()
                evaluated = (eval_value == "TRUE")
                
                if evaluated_flag and not evaluated:
                    continue

                #file_path, start_line, start_col, end_line, end_col, condition, macros_str = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)

                condition = ' '.join(condition.split())
                
                macros_key = macros_str if macros_str else ""
                entry_key = f"IF:{file_path}:{start_line}:{start_col}:{condition}:{macros_key}:active"
                if entry_key in seen_entries["if"]:
                    continue
                seen_entries["if"].add(entry_key)

                macros_info = []
                if macros_str:
                    for macro_part in macros_str.split(';'):
                        macro_match = re.match(r'(.+?)\s+defined at:\s+(.+)', macro_part.strip())
                        if macro_match:
                            m_name, m_def_loc = macro_match.groups()
                            macro_key = f"{m_name.strip()}:{m_def_loc.strip()}"

                            macros_info.append({
                                "name": m_name.strip(),
                                "definition": m_def_loc.strip(), # defined_at
                                "macro_key": macro_key,
                            })
                
                entry = {
                    "kind": "directive",
                    "type": "IF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "condition": condition,
                    "macros": macros_info,
                    "active": True,
                    "skipped": False,
                    "evaluated": evaluated
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["if"].append(entry)
                
                for macro_info in macros_info:
                    macro_name = macro_info["name"]
                    def_loc = macro_info["definition"] # defined_at
                    
                    macro_key = f"{macro_name}:{def_loc}"
                    use_entry = {
                        "kind": "directive",
                        "type": "IF",
                        "file_path": file_path,
                        "start_line": int(start_line),
                        "start_column": int(start_col),
                        "end_line": int(end_line),
                        "end_column": int(end_col),
                        "condition": condition,
                        "macro_key": macro_key,
                        "active": True,
                        "skipped": False
                    }
                    
                    """
                    if macro_key in data["macros"]:
                        data["macros"][macro_key]["uses"].append(use_entry)
                    """
                    """
                    if macro_key not in data["macros"]:
                        data["macros"][macro_key] = {
                            "name": macro_name,
                            "definitions": [],
                            "uses": []
                        }
                    """
                    def_file_path, def_line, def_col = parse_def_loc(def_loc)
                    if macro_key not in data["macros"]:
                        data["macros"][macro_key] = {
                            "name": macro_name,
                            "definition": {
                                "file_path": def_file_path,
                                "start_line": def_line,
                                "start_column": def_col
                            },
                            "is_const" : None,
                            "is_flag" : None,
                            "is_guard" : None,
                            "is_guarded" : None,
                            search_key: []
                        }
                    data["macros"][macro_key][search_key].append(use_entry)

                key = f"IF:{file_path}:{start_line}:{start_col}"
                if key not in endif_mapping:
                    endif_mapping[key] = {}
                endif_mapping[key]["file_entry"] = entry
                endif_mapping[key]["macros"] = macros_info
        
        # Parse [IF (skipped)] (inactive)
        elif line.startswith("[IF (skipped)]"):
            if skipped_flag:  # ★ Changed to skipped_flag
                continue

            if evaluated_flag:
                continue

            match = re.match(r'\[IF \(skipped\)\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s+\[(.+?)\]\s*)?$', line)
            if match:
                file_path, start_line, start_col, end_line, end_col, condition, macros_str = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                condition = ' '.join(condition.split())
                
                macros_key = macros_str if macros_str else ""
                entry_key = f"IF:{file_path}:{start_line}:{start_col}:{condition}:{macros_key}:skipped"
                if entry_key in seen_entries["if"]:
                    continue
                seen_entries["if"].add(entry_key)

                macros_info = []
                if macros_str:
                    for macro_part in macros_str.split(';'):
                        macro_match = re.match(r'(.+?)\s+defined at:\s+(.+)', macro_part.strip())
                        if macro_match:
                            m_name, m_def_loc = macro_match.groups()
                            macro_key = f"{m_name.strip()}:{m_def_loc.strip()}"

                            macros_info.append({
                                "name": m_name.strip(),
                                "definition": m_def_loc.strip(), # defined_at
                                "macro_key": macro_key,
                            })
                
                entry = {
                    "kind": "directive",
                    "type": "IF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "condition": condition,
                    "macros": macros_info,
                    "active": False,
                    "skipped": True,
                    "evaluated": False
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["if"].append(entry)
                
                for macro_info in macros_info:
                    macro_name = macro_info["name"]
                    def_loc = macro_info["definition"] # defined_at
                    
                    macro_key = f"{macro_name}:{def_loc}"
                    use_entry = {
                        "kind": "directive",
                        "type": "IF",
                        "file_path": file_path,
                        "start_line": int(start_line),
                        "start_column": int(start_col),
                        "end_line": int(end_line),
                        "end_column": int(end_col),
                        "condition": condition,
                        "macro_key": macro_key,
                        "active": False,
                        "skipped": True
                    }
                    
                    """
                    if macro_key in data["macros"]:
                        data["macros"][macro_key]["uses"].append(use_entry)
                    """
                    """
                    if macro_key not in data["macros"]:
                        data["macros"][macro_key] = {
                            "name": macro_name,
                            "definitions": [],
                            "uses": []
                        }
                    """
                    def_file_path, def_line, def_col = parse_def_loc(def_loc)
                    if macro_key not in data["macros"]:
                        data["macros"][macro_key] = {
                            "name": macro_name,
                            "definition": {
                                "file_path": def_file_path,
                                "start_line": def_line,
                                "start_column": def_col
                            },
                            "is_const" : None,
                            "is_flag" : None,
                            "is_guard" : None,
                            "is_guarded" : None,
                            search_key: []
                        }
                    data["macros"][macro_key][search_key].append(use_entry)

                key = f"IF:{file_path}:{start_line}:{start_col}"
                # if key not in endif_mapping:
                #     endif_mapping[key] = {}
                # endif_mapping[key]["file_entry"] = entry
                # endif_mapping[key]["macros"] = macros_info
        
        # Parse [ELIF] (active)
        #elif line.startswith("[ELIF]") and not line.startswith("[ELIF (skipped)]"):
        elif re.match(r'\[ELIF(?:_TRUE|_FALSE|_NOT_EVALUATED)?\]', line) and not line.startswith("[ELIF (skipped)]"):
            #match = re.match(r'\[ELIF\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s+\[(.+?)\]\s*)?$', line)
            match = re.match(r'\[ELIF(?:_(TRUE|FALSE|NOT_EVALUATED))?\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s+\[(.+?)\]\s*)?$', line)

            if match:
                #file_path, start_line, start_col, end_line, end_col, condition, macros_str = match.groups()
                eval_value, file_path, start_line, start_col, end_line, end_col, condition, macros_str = match.groups()
                evaluated = (eval_value == "TRUE")
                
                if evaluated_flag and not evaluated:
                    continue

                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                condition = ' '.join(condition.split())
                
                macros_key = macros_str if macros_str else ""
                entry_key = f"ELIF:{file_path}:{start_line}:{start_col}:{condition}:{macros_key}:active"
                if entry_key in seen_entries["elif"]:
                    continue
                seen_entries["elif"].add(entry_key)

                macros_info = []
                if macros_str:
                    for macro_part in macros_str.split(';'):
                        macro_match = re.match(r'(.+?)\s+defined at:\s+(.+)', macro_part.strip())
                        if macro_match:
                            m_name, m_def_loc = macro_match.groups()
                            macro_key = f"{m_name.strip()}:{m_def_loc.strip()}"

                            macros_info.append({
                                "name": m_name.strip(),
                                "definition": m_def_loc.strip(), # defined_at
                                "macro_key": macro_key
                            })
                
                entry = {
                    "kind": "directive",
                    "type": "ELIF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "condition": condition,
                    "macros": macros_info,
                    "active": True,
                    "skipped": False,
                    "evaluated": evaluated
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["elif"].append(entry)

                # ★ Added from here
                for macro_info in macros_info:
                    macro_name = macro_info["name"]
                    def_loc = macro_info["definition"] # defined_at
                    macro_key = f"{macro_name}:{def_loc}"
                    
                    use_entry = {
                        "kind": "directive",
                        "type": "ELIF",
                        "file_path": file_path,
                        "start_line": int(start_line),
                        "start_column": int(start_col),
                        "end_line": int(end_line),
                        "end_column": int(end_col),
                        "condition": condition,
                        "macro_key": macro_key,
                        "active": True,
                        "skipped": False
                    }
                    
                    def_file_path, def_line, def_col = parse_def_loc(def_loc)
                    if macro_key not in data["macros"]:
                        data["macros"][macro_key] = {
                            "name": macro_name,
                            "definition": {
                                "file_path": def_file_path,
                                "start_line": def_line,
                                "start_column": def_col
                            },
                            "is_const" : None,
                            "is_flag" : None,
                            "is_guard" : None,
                            "is_guarded" : None,
                            search_key: []
                        }
                    data["macros"][macro_key][search_key].append(use_entry)
                # ★ Added up to here
        

        # Parse [ELIF (skipped)] (inactive)
        elif line.startswith("[ELIF (skipped)]"):
            if skipped_flag:  # ★ Changed to skipped_flag
                continue

            if evaluated_flag:
                continue

            match = re.match(r'\[ELIF \(skipped\)\] (.+?):(\d+):(\d+):(\d+):(\d+) - (.+?)(?:\s+\[(.+?)\]\s*)?$', line)
            if match:
                file_path, start_line, start_col, end_line, end_col, condition, macros_str = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)

                condition = ' '.join(condition.split())
                
                macros_key = macros_str if macros_str else ""
                entry_key = f"ELIF:{file_path}:{start_line}:{start_col}:{condition}:{macros_key}:skipped"
                if entry_key in seen_entries["elif"]:
                    continue
                seen_entries["elif"].add(entry_key)

                macros_info = []
                if macros_str:
                    for macro_part in macros_str.split(';'):
                        macro_match = re.match(r'(.+?)\s+defined at:\s+(.+)', macro_part.strip())
                        if macro_match:
                            m_name, m_def_loc = macro_match.groups()
                            macro_key = f"{m_name.strip()}:{m_def_loc.strip()}"

                            macros_info.append({
                                "name": m_name.strip(),
                                "definition": m_def_loc.strip(), # defined_at
                                "macro_key": macro_key
                            })
                
                entry = {
                    "kind": "directive",
                    "type": "ELIF",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "condition": condition,
                    "macros": macros_info,
                    "active": False,
                    "skipped": True,
                    "evaluated": False
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][file_path]["elif"].append(entry)

                # ★ Added from here
                for macro_info in macros_info:
                    macro_name = macro_info["name"]
                    def_loc = macro_info["definition"] # defined_at
                    macro_key = f"{macro_name}:{def_loc}"
                    
                    use_entry = {
                        "kind": "directive",
                        "type": "ELIF",
                        "file_path": file_path,
                        "start_line": int(start_line),
                        "start_column": int(start_col),
                        "end_line": int(end_line),
                        "end_column": int(end_col),
                        "condition": condition,
                        "macro_key": macro_key,
                        "active": False,
                        "skipped": True
                    }
                    
                    def_file_path, def_line, def_col = parse_def_loc(def_loc)
                    if macro_key not in data["macros"]:
                        data["macros"][macro_key] = {
                            "name": macro_name,
                            "definition": {
                                "file_path": def_file_path,
                                "start_line": def_line,
                                "start_column": def_col
                            },
                            "is_const" : None,
                            "is_flag" : None,
                            "is_guard" : None,
                            "is_guarded" : None,
                            search_key: []
                        }
                    data["macros"][macro_key][search_key].append(use_entry)
                # ★ Added up to here
        
        # Parse [ELSE_TRUE] (active)
        elif line.startswith("[ELSE_TRUE]"):
            match = re.match(r'\[ELSE_TRUE\] (.+?):(\d+):(\d+):(\d+):(\d+)', line)
            if match:
                file_path, start_line, start_col, end_line, end_col = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)
                
                entry_key = f"ELSE:{file_path}:{start_line}:{start_col}:active"
                if entry_key in seen_entries.get("else", set()):
                    continue
                if "else" not in seen_entries:
                    seen_entries["else"] = set()
                seen_entries["else"].add(entry_key)

                entry = {
                    "kind": "directive",
                    "type": "ELSE",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start": int(start_line),
                    "block_end": int(end_line),
                    "active": True,
                    "skipped": False,
                    "evaluated": True
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else": [],
                        "endif": []
                    }
                
                if "else" not in data["files"][file_path]:
                    data["files"][file_path]["else"] = []
                
                data["files"][file_path]["else"].append(entry)
        
        # Parse [ELSE_FALSE] (skipped)
        elif line.startswith("[ELSE_FALSE]"):
            if skipped_flag:      # ★ Added
                continue

            if evaluated_flag:
                continue  # evaluated=False so skip

            match = re.match(r'\[ELSE_FALSE\] (.+?):(\d+):(\d+):(\d+):(\d+)', line)
            if match:
                file_path, start_line, start_col, end_line, end_col = match.groups()
                file_path = get_abs_path(file_path)
                file_path = put_in_target_dir(file_path, target_dir)

                entry_key = f"ELSE:{file_path}:{start_line}:{start_col}:skipped"
                if entry_key in seen_entries.get("else", set()):
                    continue
                if "else" not in seen_entries:
                    seen_entries["else"] = set()
                seen_entries["else"].add(entry_key)

                entry = {
                    "kind": "directive",
                    "type": "ELSE",
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start": int(start_line),
                    "block_end": int(end_line),
                    "active": False,
                    "skipped": True,
                    "evaluated": False
                }
                
                if file_path not in data["files"]:
                    data["files"][file_path] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else": [],
                        "endif": []
                    }
                
                if "else" not in data["files"][file_path]:
                    data["files"][file_path]["else"] = []
                
                data["files"][file_path]["else"].append(entry)
        
        
        
        # Parse [ENDIF] (active & skipped)
        elif line.startswith("[ENDIF]"):
            is_skipped = "(skipped)" in line
            
            if skipped_flag and is_skipped:  # ★ Changed to skipped_flag
                continue

            if evaluated_flag and is_skipped:
                continue  # ★ Added this line

            match = re.match(r'\[ENDIF(?: \(skipped\))?\] (.+?):(\d+):(\d+):(\d+):(\d+)', line)
            if match:
                endif_file, start_line, start_col, end_line, end_col = match.groups()
                endif_file = get_abs_path(endif_file)
                
                status = "skipped" if is_skipped else "active"
                entry_key = f"ENDIF:{endif_file}:{start_line}:{start_col}:{status}"
                if entry_key in seen_entries["endif"]:
                    continue
                seen_entries["endif"].add(entry_key)

                entry = {
                    "kind": "directive",
                    "type": "ENDIF",
                    "file_path": endif_file,
                    "start_line": int(start_line),
                    "start_column": int(start_col),
                    "end_line": int(end_line),
                    "end_column": int(end_col),
                    "block_start" : int(start_line),
                    "block_end" : int(end_line),
                    "closes": None,
                    "active": not is_skipped,
                    "skipped": is_skipped
                }
                
                if endif_file not in data["files"]:
                    data["files"][endif_file] = {
                        "defined": [],
                        "ifdef": [],
                        "ifndef": [],
                        "if": [],
                        "elif": [],
                        "else" : [],
                        "endif": []
                    }
                
                data["files"][endif_file]["endif"].append(entry)
                pending_endif = entry
        
        # Parse "=> Closes [IF/IFDEF/IFNDEF]"
        elif line.startswith("=> Closes"):
            
            match = re.match(r'=> Closes \[(\w+(?:\s+\(skipped\))?)\] at (.+?):(\d+):(\d+):(\d+):(\d+)(?: \((.+?)\))?$', line)
            match = re.match(r'=> Closes \[(\w+(?:\s+\(skipped\))?)\] at (.+?):(\d+):(\d+):(\d+):(\d+)(?:\s*\(\s*(.+?)\s*\))?$', line)
            match = re.match(r'=> Closes \[(\w+(?:_TRUE|_FALSE|_FUNC_TRUE|_FUNC_FALSE|_NOT_EVALUATED)?(?:\s+\(skipped\))?)\] at (.+?):(\d+):(\d+):(\d+):(\d+)(?:\s*\(\s*(.+?)\s*\))?$', line)

            if match:
                if_type, if_file, start_line, start_col, end_line, end_col, if_info = match.groups()
                if_file = get_abs_path(if_file)

                # if_type, if_file, start_line, start_col, end_line, end_col, if_info = match.groups()
                # if_file = get_abs_path(if_file)
                
                is_skipped = "(skipped)" in if_type
                if_type = re.sub(r'_(TRUE|FALSE|FUNC_TRUE|FUNC_FALSE|NOT_EVALUATED)', '', if_type)
                if_type = if_type.replace(" (skipped)", "")

                # if_type = if_type.replace(" (skipped)", "")
                
                if if_info:
                    if_info = ' '.join(if_info.split())

                # if pending_endif is None:
                #     continue
                
                lookup_key = f"{if_type}:{if_file}:{start_line}:{start_col}"
                
                # Create the key if it doesn't exist
                if lookup_key not in endif_mapping:
                    endif_mapping[lookup_key] = {}

                endif_file, e_start_line, e_start_col, e_end_line, e_end_col, is_endif_skipped = get_endif_info(processed_lines, idx, line)

                # Set closes information
                
                if lookup_key not in endif_mapping:
                    endif_mapping[lookup_key] = {}

                endif_mapping[lookup_key] = {
                    "kind": "directive",
                    "type": if_type,
                    "def_file_path": if_file,
                    "def_start_line": int(start_line),
                    "def_start_column": int(start_col),
                    "def_end_line": int(end_line),
                    "def_end_column": int(end_col),
                    "file_path": endif_file,
                    "start_line": int(e_start_line),
                    "start_column": int(e_start_col),
                    "end_line": int(e_end_line),
                    "end_column": int(e_end_col),
                    #"info": None, #if_info if if_info else "",
                    "skipped": is_endif_skipped
                }

                # endif_mapping[lookup_key]["endif_info"] = {
                #     "type": if_type,
                #     "file_path": endif_file,
                #     "start_line": int(e_start_line),
                #     "start_column": int(e_start_col),
                #     "end_line": int(e_end_line),
                #     "end_column": int(e_end_col),
                #     "info": None, #if_info if if_info else "",
                #     "skipped": is_endif_skipped
                # }
                
                # endif_mapping[lookup_key]["endif_info"] = {
                #     "file_path": pending_endif["file_path"],
                #     "start_line": pending_endif["start_line"],
                #     "start_column": pending_endif["start_column"],
                #     "end_line": pending_endif["end_line"],
                #     "end_column": pending_endif["end_column"]
                # }
                
                # # Set closes information
                # pending_endif["closes"] = {
                #     "type": if_type,
                #     "file_path": if_file,
                #     "start_line": int(start_line),
                #     "start_column": int(start_col),
                #     "end_line": int(end_line),
                #     "end_column": int(end_col),
                #     "info": if_info if if_info else "",
                #     "skipped": is_skipped
                # }
                
                pending_endif = None


    ################
    # Insert endif information all at once at the end
    ################
    
    write_json(f"{database_dir}/endif_mapping.json", endif_mapping)

    ################
    # Insert endif information all at once at the end
    ################
    
    # added
    # ★ Build index for fast lookup of uses
    use_index = {}
    for macro_key, macro_data in data["macros"].items():
        for use in macro_data.get(search_key, []):
            idx_key = (use["type"], use["file_path"], use["start_line"], use["start_column"])
            if idx_key not in use_index:
                use_index[idx_key] = []
            use_index[idx_key].append(use)
    # ended

    # ★ Also build index for fast lookup of files
    # file_index = {}
    # for file_path, file_data in data["files"].items():
    #     for directive_type in ["ifdef", "ifndef", "if", "elif", "else"]:
    #         for entry in file_data.get(directive_type, []):
    #             idx_key = (directive_type, file_path, entry["start_line"], entry["start_column"])
    #             file_index[idx_key] = entry
    
    # Changed: keep all entries with the same key as a list
    file_index_all = {}
    for file_path, file_data in data["files"].items():
        for directive_type in ["ifdef", "ifndef", "if", "elif", "else"]:
            for entry in file_data.get(directive_type, []):
                idx_key = (directive_type, file_path, entry["start_line"], entry["start_column"])
                if idx_key not in file_index_all:
                    file_index_all[idx_key] = []
                file_index_all[idx_key].append(entry)  # ← Both are kept
                
    updated_count = 0
    for lookup_key, endif_data in endif_mapping.items():
        # Create endif_info from endif_data
        endif_info = {
            "file_path": endif_data["file_path"],
            "start_line": endif_data["start_line"],
            "start_column": endif_data["start_column"],
            "end_line": endif_data["end_line"],
            "end_column": endif_data["end_column"]
        }
        
        # Extract type, file_path, start_line, start_col from lookup_key
        # Format: "IF:/path/to/file.c:54:2"
        parts = lookup_key.split(":")
        directive_type = parts[0]  # IF, IFDEF, IFNDEF, etc.
        def_file_path = endif_data["def_file_path"]
        def_start_line = endif_data["def_start_line"]
        def_start_col = endif_data["def_start_column"]
        

        # Find the corresponding entry in data["files"] and add endif
        """
        if def_file_path in data["files"]:
            # Search the list corresponding to directive_type
            directive_list_name = directive_type.lower()  # "IF" -> "if"
            if directive_list_name in data["files"][def_file_path]:
                for entry in data["files"][def_file_path][directive_list_name]:
                    if (entry["start_line"] == def_start_line and 
                        entry["start_column"] == def_start_col):
                        entry["endif"] = endif_info
                        updated_count += 1
                        
        """
        # ★ files: O(1) lookup
        # file_idx_key = (directive_type.lower(), def_file_path, def_start_line, def_start_col)
        # entry = file_index.get(file_idx_key)
        # if entry:
        #     entry["endif"] = endif_info
        #     updated_count += 1

        # Changed: associate with all entries at the same position
        file_idx_key = (directive_type.lower(), def_file_path, def_start_line, def_start_col)
        entries = file_index_all.get(file_idx_key, [])
        for entry in entries:
            entry["endif"] = endif_info
            updated_count += 1

        """
        # Also add endif to uses in data["macros"]
        for macro_key, macro_data in data["macros"].items():
            for use in macro_data.get("uses", []):
                if (use["type"] == directive_type and 
                    use["file_path"] == def_file_path and
                    use["start_line"] == def_start_line and
                    use["start_column"] == def_start_col):
                    use["endif"] = endif_info
        """
        idx_key = (directive_type, def_file_path, def_start_line, def_start_col)
        for use in use_index.get(idx_key, []):
            use["endif"] = endif_info

    print(f"✅ Updated {updated_count} entries with endif information")
    #print(f"✅ Updated {updated_count} entries with endif information")

    #######
    for macro_key, file_data in data["macros"].items():        
        # Merge ifdef, ifndef, if, elif
        for directive_type in ['ifdef', 'ifndef', 'if', 'elif', 'else']:
            for item in file_data.get(directive_type, []):
                start_line = item.get('start_line')

                if 'endif' not in item:
                    continue
                endif_info = item.get('endif', {})
                end_line = endif_info.get('start_line') # This is the key point

                item['block_end'] = end_line
    

    for file_path, file_data in data["files"].items():        
        # Merge ifdef, ifndef, if, elif
        for directive_type in ['ifdef', 'ifndef', 'if', 'elif', 'else']:
            for item in file_data.get(directive_type, []):
                start_line = item.get('start_line')

                if 'endif' not in item:
                    continue
                endif_info = item.get('endif', {})
                end_line = endif_info.get('start_line') # This is the key point

                item['block_end'] = end_line

    """
    for macro_key, file_data in data["macros"].items():        
        # Merge ifdef, ifndef, if, elif
        for directive_type in ['ifdef', 'ifndef', 'if', 'elif']:
            for item in file_data.get(directive_type, []):
                start_line = item.get('start_line')

                if 'endif' not in item:
                    continue
                endif_info = item.get('endif', {})
                end_line = endif_info.get('start_line') # This is the key point

                item['block_end'] = end_line
    """
    ################

    with open(unordered_macros_path, 'w', encoding='utf-8') as f:
        json.dump(data["files"], f, indent=4, ensure_ascii=False)
    
    print(f"Saved macro information to: {unordered_macros_path}")

    with open(macros_path, 'w', encoding='utf-8') as f:
        json.dump(data["macros"], f, indent=4, ensure_ascii=False)
    
    print(f"Saved macro information to: {macros_path}")

    total_files = len(data["files"])
    print(f"\nStatistics:")
    print(f"  Total files: {total_files}")
    
    return data


def divide_macros(unordered_taken_directive_path, taken_directive_path, meta_dir, target_dir):
    """Split output.json by file path and save to meta_dir"""
    target_dir = Path(target_dir)
    # Create metadata directory
    meta_path = Path(meta_dir)
    meta_path.mkdir(exist_ok=True)
    
    # Load JSON
    if not os.path.exists(unordered_taken_directive_path):
        #print()
        return

    with open(unordered_taken_directive_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(taken_directive_path, 'r', encoding='utf-8') as f:
        def_data = json.load(f)

    
    # # Overall map of macro information (for reference from each file)
    # macros_map_path = meta_path / "macros_all.json"
    # with open(macros_map_path, 'w', encoding='utf-8') as f:
    #     json.dump(data["macros"], f, indent=4, ensure_ascii=False)
    # print(f"Saved all macros map to: {macros_map_path}")
    
    # Save JSON for each file
    saved_count = 0
    for file_path, file_data in data.items():
        # Normalize path
        path_obj = Path(file_path)

        # Convert relative path to absolute path
        if not path_obj.is_absolute():
            path_obj = (target_dir / file_path).resolve()

        safe_name = str(path_obj).lstrip('/').replace('/', '_').replace('.', '_') + '.json'
        save_path = meta_path / safe_name
        
        # Collect macro names used in this file
        used_macro_names = set()
        used_macro_keys = set()
        
        for ifdef_entry in file_data.get("ifdef", []):
            used_macro_names.add(ifdef_entry["macro_name"])
            used_macro_keys.add(ifdef_entry["macro_key"])
        
        for ifndef_entry in file_data.get("ifndef", []):
            used_macro_names.add(ifndef_entry["macro_name"])
            used_macro_keys.add(ifndef_entry["macro_key"])
        
        for if_entry in file_data.get("if", []):
            for macro_info in if_entry.get("macros", []):
                used_macro_names.add(macro_info["name"])
                used_macro_keys.add(macro_info["macro_key"])
        
        for elif_entry in file_data.get("elif", []):
            for macro_info in elif_entry.get("macros", []):
                used_macro_names.add(macro_info["name"])
                used_macro_keys.add(macro_info["macro_key"])
        
        # Build macro information per definition location
        macros_list = []
        
        """
        for macro_name in used_macro_names:
            if macro_name not in data["macros"]:
                continue
            
            macro_data = data["macros"][macro_name]
            
            definitions = macro_data.get("definitions", [])
        """

        for macro_key in used_macro_keys:

            # This section needs overall review
            defined = True
            if macro_key not in def_data:
                macro_name = ""
                defined = False
            else:
                macro_data = def_data[macro_key]
                definitions = macro_data.get("definitions", [])
                macro_name = macro_data.get("name", [])
                if not definitions:
                    defined = False

            if defined is False: #if not definitions:
                # Case of undefined macro
                macro_entry = {
                    "key": macro_key,
                    "name": macro_name,
                    "definition": "undefined",  # ★ Set "undefined"
                    "uses": macro_data.get("uses", [])
                }
                macros_list.append(macro_entry)

            else:
                # Process each definition location
                for definition in macro_data.get("definitions", []):
                    # Get all usage locations for this definition
                    uses = macro_data.get("uses", [])
                    
                    macro_entry = {
                        "key": macro_key,
                        "name": macro_name,
                        "definition": {
                            "file_path": definition["file_path"],
                            "line": definition["line"],
                            "column": definition["column"]
                        },
                        "uses": uses
                    }
                    
                    macros_list.append(macro_entry)
        
        # Create file-specific data
        file_specific_data = {
            "source_file": file_path,
            #"directives": file_data,
            "macros": macros_list
        }
        
        # Save
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(file_specific_data, f, indent=4, ensure_ascii=False)
        
        saved_count += 1
        print(f"Saved: {save_path}")
    
    print(f"\nDivision complete:")
    print(f"  Total files saved: {saved_count}")
    print(f"  Output directory: {meta_path.absolute()}")
    
    # Create index file
    index_data = {
        "total_files": saved_count,
        "files": {}
    }
    
    for file_path in data.keys():
        path_obj = Path(file_path)
        
        if not path_obj.is_absolute():
            path_obj = (target_dir / file_path).resolve()  # ★ Use target_dir

        safe_name = str(path_obj).lstrip('/').replace('/', '_').replace('.', '_') + '.json'
        index_data["files"][file_path] = safe_name
    
    index_path = meta_path / "index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4, ensure_ascii=False)
    print(f"Saved index file: {index_path}")



def insert_header_directive(custom_header_path):
    
    with open(custom_header_path, 'r', encoding='utf-8') as f:
        content_lines = f.readlines()

    # ★ Check if header guard already exists
    has_header_guard = False
    for line in content_lines:
        if '#ifndef STATE_MACROS_H' in line or '#define STATE_MACROS_H' in line:
            has_header_guard = True
            break
    
    if has_header_guard:
        print(f"Header guard already exists in: {custom_header_path}")
        return  # ★ Already inserted, so do nothing

    custom_header_lines = [
        "// Auto-generated state macro definitions\n",
        #"// DO NOT EDIT MANUALLY\n",
        "\n",
        "#ifndef STATE_MACROS_H\n",
        "#define STATE_MACROS_H\n",
        "\n"
    ]
    
    # Append content
    custom_header_lines.extend(content_lines)
    
    # End of header guard
    custom_header_lines.extend([
        "\n",
        "#endif // STATE_MACROS_H\n"
    ])
    
    # Write to file
    with open(custom_header_path, 'w', encoding='utf-8') as f:
        f.writelines(custom_header_lines)
    
    #print(f"Created custom header: {custom_header_path}")

    

def get_each_version_path(file_path, max_version):
    """Return a list of versioned paths from the original file path"""
    base, ext = os.path.splitext(file_path)
    return [f"{base}_{i}{ext}" for i in range(1, max_version + 1)]


def normalize_repliction_path(appearances, all_merge_maps):
    """Normalize appearance paths using merge information"""
    normalized = []
    for app in appearances:
        # app = "/path/to/feature_2.h:10:5" format
        file_path, line, column = parse_def_loc(app)
        
        # If this file is a version deleted by merge, replace with merge destination
        for original_path, merge_map in all_merge_maps.items():
            for removed_path, keep_path in merge_map.items():
                if file_path == removed_path:
                    file_path = keep_path
                    break
        """
        for original_path, merge_map in all_merge_maps.items():
            base, ext = os.path.splitext(original_path)
            for removed_v, keep_v in merge_map.items():
                removed_path = f"{base}_{removed_v}{ext}"
                keep_path = f"{base}_{keep_v}{ext}"
                if file_path == removed_path:
                    file_path = keep_path
                    break
        """
        
        normalized.append(f"{file_path}:{line}:{column}")
    
    return tuple(sorted(normalized))


def rewrite_includes_for_merge(target_dir, all_merge_maps):
    """Scan all files and rewrite #includes pointing to deleted merge versions
    to #includes pointing to the representative version"""
    
    # Build a conversion table: removed path -> representative path
    replace_map = {}  # {removed_filename: keep_filename}
    """
    for original_path, merge_map in all_merge_maps.items():
        base, ext = os.path.splitext(original_path)
        for removed_v, keep_v in merge_map.items():
            removed_name = os.path.basename(f"{base}_{removed_v}{ext}")
            keep_name = os.path.basename(f"{base}_{keep_v}{ext}")
            replace_map[removed_name] = keep_name
    """

    for original_path, merge_map in all_merge_maps.items():
        for removed_path, keep_path in merge_map.items():
            removed_name = os.path.basename(removed_path)
            keep_name = os.path.basename(keep_path)
            replace_map[removed_name] = keep_name
            
    if not replace_map:
        return
    
    # Pattern for #include lines
    include_pattern = re.compile(r'(#\s*include\s*[""<])([^"">]+)(["">>])')
    
    for root, dirs, files in os.walk(target_dir):
        for fname in files:
            if not fname.endswith(('.h', '.c')):
                continue
            fpath = os.path.join(root, fname)
            
            with open(fpath, 'r') as f:
                content = f.read()
            
            new_content = content
            for line in content.splitlines():
                m = include_pattern.match(line.strip())
                if m:
                    included_file = os.path.basename(m.group(2))
                    if included_file in replace_map:
                        old_path = m.group(2)
                        new_path = old_path.replace(included_file, replace_map[included_file])
                        new_content = new_content.replace(
                            f"{m.group(1)}{old_path}{m.group(3)}",
                            f"{m.group(1)}{new_path}{m.group(3)}"
                        )
            
            if new_content != content:
                with open(fpath, 'w') as f:
                    f.write(new_content)
                print(f"  Rewrote includes in: {fpath}")



def merge_redundant_headers(include_chain, versions, target_dir, database_dir, div_meta_dir):
    
    all_merge_maps = {}  # Accumulate merge results
    merged_flag = False

    for file_path in include_chain:
        max_version = versions.get(file_path, 1)

        if max_version <= 1:
            continue
        
        print(f"\nChecking: {file_path} ({max_version} versions)")

        version_paths = get_each_version_path(file_path, max_version)
        
        version_signatures = {}
        
        for each_path in version_paths:
            meta_data, meta_path = obtain_metadata(each_path, div_meta_dir, False, None, "def")
            if meta_data is None: # There may be cases where no number is assigned
                continue
            
            #print(meta_path)
            sig = {}
            for key, item in meta_data.items():
                if 'kind' in item and "macro" in item['kind']: # == "macro":
                    definition = item['definition']
                    normalized_def = normalize_repliction_path(
                        [definition],
                        all_merge_maps
                    )
                    appearances = normalize_repliction_path(
                        item.get('appearances', []),
                        all_merge_maps
                    )
                    sig[item['name']] = (normalized_def[0], appearances)

                if 'kind' in item and item['kind'] == "directive": 
                    if 'definition' in item:
                        item_file = item['file_path']
                        item_line = item['start_line']
                        item_column = item['start_column']
                        #print(meta_path)
                        definition = item['definition'] # defined_at

                        normalized_def = normalize_repliction_path(
                            [definition],
                            all_merge_maps
                        )

                        if item_line is not None:
                            appearances = normalize_repliction_path(
                                [f"{item_file}:{item_line}:{item_column}"],
                                all_merge_maps
                            )
                            sig[item['name']] = (normalized_def[0], appearances)
                        
                    else:
                        if 'macros' not in item:
                            continue

                        for each_item in item['macros']:
                            if 'definition' in each_item:
                                #print(meta_path)
                                definition = each_item['definition'] # defined_at
                                item_file, item_line, item_column = parse_def_loc(definition)

                                normalized_def = normalize_repliction_path(
                                    [definition],
                                    all_merge_maps
                                )
                                
                                if item_line is not None:
                                    appearances = normalize_repliction_path(
                                        [f"{item_file}:{item_line}:{item_column}"],
                                        all_merge_maps
                                    )
                                    sig[each_item['name']] = (normalized_def[0], appearances)
                                    

            version_signatures[each_path] = sig
        
        # Detect versions with identical signatures
        paths = list(version_signatures.keys())
        merge_map = {}  # {removed_version: keep_version}
        merged = set()
        
        for i in range(len(paths)):
            if paths[i] in merged:
                continue
            for j in range(i + 1, len(paths)):
                if paths[j] in merged:
                    continue
                if version_signatures[paths[i]] == version_signatures[paths[j]]:
                    print(f"  Redundant: {paths[j]} == {paths[i]}")
                    merged.add(paths[j])
                    # Extract version numbers and record in merge_map
                    # TODO: Deletion and rewriting of parent's #include
                    merge_map[paths[j]] = paths[i]
        
        if merged:
            all_merge_maps[file_path] = merge_map

    
    write_json(f"{database_dir}/merge_maps.json", all_merge_maps)

    ###
    delete_flag = False #True #False
    # Actual merge work
    if all_merge_maps:
        # Rewrite parent file's #include to the representative version
        rewrite_includes_for_merge(target_dir, all_merge_maps)
        
        # Delete redundant files if the flag is enabled
        if delete_flag:
            delete_merged_files(target_dir, all_merge_maps)

    ###
    if all_merge_maps:
        merged_flag = True

    return merged_flag  #all_merge_maps



def delete_merged_files(target_dir, all_merge_maps):

    for original_path, merge_map in all_merge_maps.items():
        base, ext = os.path.splitext(original_path)
        for removed_v, keep_v in merge_map.items():
            removed_path = f"{base}_{removed_v}{ext}"
            if os.path.exists(removed_path):
                os.remove(removed_path)
                print(f"  Deleted: {removed_path}")



def detect_conditioned_macros(taken_directive_path, output_path):
    # Here we actually obtain the macro variable names that are purely used conditionally
    # (this might be wrong though)

    """
    Detect macros defined inside conditional compilation.
    If a macro definition is inside a conditional block, record all macros that set that condition.
    """
    print("Detecting conditioned macros...")
    
    #meta_dir = Path(meta_dir)
    
    # Phase 1: Collect all conditional blocks
    print("Phase 1: Collecting all conditional blocks...")
    all_conditional_blocks = []
    
    with open(taken_directive_path, 'r') as f:
        data = json.load(f)

    """
    for meta_file in meta_dir.rglob("*.json"):
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            
            macros = data.get('macros', [])
    """
    for macro_key, macro in data.items():
        condition_macro_name = macro.get('name')
        uses = macro.get('uses', [])
        
        for use in uses:
            use_type = use.get('type')
            if use_type not in ['IFDEF', 'IFNDEF', 'IF']:
                continue
            
            use_file = use.get('file_path')
            use_line = use.get('line')
            use_column = use.get('column')
            endif = use.get('endif', {})
            endif_file = endif.get('file_path')
            endif_line = endif.get('line')
            endif_column = endif.get('column')
            
            if not (use_file and use_line and endif_file and endif_line):
                continue
            
            all_conditional_blocks.append({
                'condition_macro': condition_macro_name,  # Macro name used in the condition
                'type': use_type,
                'file_path': use_file,
                'start_line': use_line,
                'start_column': use_column,
                'end_line': endif_line,
                'end_column': endif_column,
                'span': endif_line - use_line
            })

        # except Exception as e:
        #     print(f"Error reading {meta_file} in phase 1: {e}")
        #     continue
    
    print(f"Found {len(all_conditional_blocks)} conditional blocks")
    
    # Phase 2: Check whether each macro definition is inside a conditional block
    print("Phase 2: Checking macro definitions...")
    conditioned_macros = []
    
    for meta_file in meta_dir.rglob("*.json"):
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            
            source_file = data.get('file_path')
            macros = data.get('macros', [])
            
            for macro in macros:
                name = macro.get('name')
                definition = macro.get('definition')
                
                # Process only when definition is dict type (has definition location info)
                if not isinstance(definition, dict):
                    continue
                
                def_file = definition.get('file_path')
                def_line = definition.get('line')
                def_column = definition.get('column')
                
                if not (def_file and def_line):
                    continue
                
                # Find all conditional blocks enclosing this definition
                enclosing_conditions = []
                
                for block in all_conditional_blocks:
                    # Check if the definition location is inside the conditional block
                    if (def_file == block['file_path'] and
                        block['start_line'] < def_line < block['end_line']):
                        
                        enclosing_conditions.append(block.copy())
                
                # If defined inside a conditional block
                if enclosing_conditions:
                    # Get the outermost conditional block (largest span)
                    outermost = max(enclosing_conditions, key=lambda x: x['span'])
                    
                    # Add to the list
                    conditioned_macros.append({
                        'name': name,
                        'file_path': source_file,
                        'definition': {
                            'file_path': def_file,
                            'line': def_line,
                            'column': def_column
                        },
                        'outermost_condition': outermost,
                        'all_conditions': enclosing_conditions,
                        'nesting_level': len(enclosing_conditions),
                        'conditioning_macros': list(set([c['condition_macro'] for c in enclosing_conditions]))  # All macro names used in conditions
                    })
        
        except Exception as e:
            print(f"Error reading {meta_file} in phase 2: {e}")
            continue
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(conditioned_macros, f, indent=4)
    
    print(f"\nFound {len(conditioned_macros)} conditioned macro definitions")
    
    # Display statistics
    nesting_stats = defaultdict(int)
    macro_name_counts = defaultdict(int)
    conditioning_macro_stats = defaultdict(int)
    
    for macro_info in conditioned_macros:
        nesting_stats[macro_info['nesting_level']] += 1
        macro_name_counts[macro_info['name']] += 1
        
        # Statistics of macros used in conditions
        for cond_macro in macro_info['conditioning_macros']:
            conditioning_macro_stats[cond_macro] += 1
    
    print("\nNesting level statistics:")
    for level in sorted(nesting_stats.keys()):
        print(f"  Level {level}: {nesting_stats[level]} definitions")
    
    # Statistics of macros with the same name
    duplicate_names = {name: count for name, count in macro_name_counts.items() if count > 1}
    if duplicate_names:
        print(f"\nFound {len(duplicate_names)} macro names with multiple definitions:")
        for name, count in sorted(duplicate_names.items(), key=lambda x: -x[1])[:5]:
            print(f"  {name}: {count} definitions")
    
    # Conditioning macro statistics
    print(f"\nTop conditioning macros (used in conditions):")
    for macro, count in sorted(conditioning_macro_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {macro}: conditions {count} definitions")
    
    #print(f"\nSaved to: {output_path}")
    
    return conditioned_macros


def get_headers(dep_json_path, target_dir):

    headers = []

    with open(dep_json_path, 'r') as f:
        data = json.load(f)
    
    # Use a set to collect unique header paths
    header_set = set()
    
    for item in data:
        # Get the source file
        source = item.get('source', '')
        # if not source.startswith(os.path.abspath(target_dir)):
        #     continue
        
        """
        # Check if source is a header file (.h, .hpp, .hh, etc.)
        if source and is_header_file(source):
            header_set.add(source)
        """
        
        # Get files from 'include' list
        includes = item.get('include', [])
        for inc in includes:
            # #if is_header_file(inc):
            # if not source.startswith(os.path.abspath(target_dir)):
            #     continue

            header_set.add(inc)
        
        """
        # Get files from 'indirect_include' list
        indirect_includes = item.get('indirect_include', [])
        for inc in indirect_includes:
            if is_header_file(inc):
                header_set.add(inc)
        """
    
    # Convert set to sorted list
    headers = sorted(list(header_set))
    
    print(f"Found {len(headers)} unique header file(s)")
    
    return headers


def parse_header(header):
    parts = header.rsplit(":", 2)
    if len(parts) != 3:
        return header

    file_path = parts[0]
    line = parts[1]
    column = parts[2]

    return file_path


def generate_header_paths_rust_code(headers):
    """
    Generate Rust code for header paths vector from Python list
    
    Args:
        headers: Python list of header file paths
        
    Returns:
        String containing Rust code for the vec! initialization
    """
    if not headers:
        # Return empty vec with same formatting as template
        return '    let config_paths = vec![        \n    ];'
    
    # Format each header path as a Rust string literal with proper indentation
    rust_lines = ['    let config_paths = vec![']
    
    for header in headers:
        parsed_header = parse_header(header)
        rust_lines.append(f'        "{parsed_header}",') #header}",')
    
    rust_lines.append('    ];')
    
    return '\n'.join(rust_lines)




def get_entry_points(target_dir, is_program_path):
    """
    Get entry points from compile_commands.json
    """
    compile_commands_dir = find_compile_commands_json(target_dir)
    compile_commands_path = os.path.join(compile_commands_dir, "compile_commands.json")

    with open(compile_commands_path, 'r') as f:
        data = json.load(f)
    
    print(compile_commands_path)
    #print(is_program_path)

    """
    target_dir_abs = os.path.abspath(target_dir)
    if not target_dir_abs.endswith('/'):
        target_dir_abs += '/'
    """
    
    entry_points = []
    program_files = set(read_json(is_program_path))
    
    for item in data:
        file_path = item.get('file', '')
        # Only files within target_dir
        if not is_system_file(file_path, program_files):
        #if file_path.startswith(target_dir_abs):
            entry_points.append(file_path)
    
    # print(compile_commands_path)
    # print(entry_points)

    entry_points = list(set(entry_points))
    return entry_points


def generate_cargo_toml(toml_path):
    """
    Update Cargo.toml while preserving comments and formatting
    Uses tomlkit which preserves the original formatting
    
    Args:
        toml_path: Path to the Cargo.toml file
    """
    toml_path = Path(toml_path)
    
    # Read existing Cargo.toml
    with open(toml_path, 'r') as f:
        content = f.read()
        cargo_config = tomlkit.parse(content)

    print(f"Updating {toml_path} (preserving format)...")
    
    # Ensure [build-dependencies] section exists
    if 'build-dependencies' not in cargo_config:
        cargo_config['build-dependencies'] = tomlkit.table()
    
    build_deps = cargo_config['build-dependencies']
    
    # Add/update required build dependencies
    if 'bindgen' not in build_deps:
        build_deps['bindgen'] = "0.72"
        print("  Added: bindgen")
    
    if 'syn' not in build_deps:
        syn_table = tomlkit.inline_table()
        syn_table['version'] = "2.0"
        syn_table['features'] = ["full", "parsing"]
        build_deps['syn'] = syn_table
        print("  Added: syn")
    
    if 'quote' not in build_deps:
        build_deps['quote'] = "1.0"
        print("  Added: quote")

    ##
    if 'serde' not in build_deps:
        serde_table = tomlkit.inline_table()
        serde_table['version'] = "1"
        serde_table['features'] = ["derive"]
        build_deps['serde'] = serde_table
        print("  Added: serde")

    if 'serde_json' not in build_deps:
        build_deps['serde_json'] = "1"
        print("  Added: serde_json")
    ##
    
    # Ensure [lib] section exists
    if 'lib' not in cargo_config:
        cargo_config['lib'] = tomlkit.table()
    
    # Add/update crate-type
    if 'crate-type' not in cargo_config['lib']:
        cargo_config['lib']['crate-type'] = ["staticlib"]
        print("  Added: lib.crate-type = [\"staticlib\"]")
    
    # Write back to file
    with open(toml_path, 'w') as f:
        f.write(tomlkit.dumps(cargo_config))
    print(f"Successfully updated {toml_path}")



def generate_run_all_path(run_all_path, run_all_template, target):
    """
    Generate run_all.sh script by replacing {target_name} with target
    
    Args:
        run_all_path: Path where the output script will be saved
        run_all_template: Path to the template file or template string
        target: Target name to replace {target_name} with
    """
    print("Generating run_all_path...")
    
    # It's a file path
    with open(run_all_template, 'r') as f:
        template_content = f.read()

    # Replace {target_name} with target
    output_content = template_content.replace('{target_name}', target)
    
    # Write to output file
    with open(run_all_path, 'w') as f:
        f.write(output_content)
    
    # Make the script executable
    os.chmod(run_all_path, 0o755)
    
    print(f"Successfully generated {run_all_path}")
    


# Extract all define locations of static macros -> define in build.rs
def extract_macro_def(meta_dir, dynamic_path, conditioned_path, build_rs_path):
    """
    Extract all define locations of static macros and generate build.rs
    
    Step 1: Get pure macros that are not determined by conditions
    Step 2: Write the condition parts of conditioned macros
    // Set cfg flags in build.rs based on macro values
    if name == "A" && value + 1 > 2 {
        println!("cargo:rustc-cfg=condition_met");
    }

    # After that, fix compile errors with LLM
    """
    print("Extracting macro definitions for build.rs...")
    
    meta_dir = Path(meta_dir)
    
    # Load dynamic_macros.json and conditioned_macros.json
    with open(dynamic_path, 'r') as f:
        dynamic_macros = json.load(f)
    
    with open(conditioned_path, 'r') as f:
        conditioned_macros = json.load(f)
    
    # Step 1: Collect pure macros (neither dynamic nor conditional)
    pure_macros = {}
    conditioned_macro_keys = set()
    
    # Create a list of conditional macros (identified by name+file+line)
    for cond_macro in conditioned_macros:
        key = (
            cond_macro['name'],
            cond_macro['definition']['file_path'],
            cond_macro['definition']['line']
        )
        conditioned_macro_keys.add(key)
    
    # Collect all macro definitions
    for meta_file in meta_dir.rglob("*.json"):
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            
            macros = data.get('macros', [])
            
            for macro in macros:
                name = macro.get('name')
                definition = macro.get('definition')
                
                # Skip if no definition location info
                if not isinstance(definition, dict):
                    continue
                
                def_file = definition.get('file_path')
                def_line = definition.get('line')
                
                # Exclude dynamic macros
                if name in dynamic_macros:
                    continue
                
                # Exclude conditional macros
                key = (name, def_file, def_line)
                if key in conditioned_macro_keys:
                    continue
                
                # Record as pure macro (need to get the actual value)
                # Here we only record the macro name and definition location
                if name not in pure_macros:
                    pure_macros[name] = []
                
                pure_macros[name].append({
                    'file_path': def_file,
                    'line': def_line,
                    'column': definition.get('column')
                })
        
        except Exception as e:
            print(f"Error reading {meta_file}: {e}")
            continue
    
    print(f"Found {len(pure_macros)} pure macro names")
    print(f"Found {len(conditioned_macros)} conditioned macro definitions")
    
    # Step 2: Generate build.rs
    generate_build_rs(pure_macros, conditioned_macros, build_rs_path)
    
    
    # return {
    #     'pure_macros': pure_macros,
    #     'conditioned_macros': conditioned_macros
    # }


# ★ insert_individual_header_include with debug output added
def insert_custom_header_include(lines, first_use_line, source_file_path, header_file_path):
    """
    Insert include for an individual header file (debug version)
    """
    # print(f"  [DEBUG] Attempting to insert include:")
    # print(f"    Header: {header_file_path}")
    # print(f"    Source: {source_file_path}")
    # print(f"    Line: {first_use_line}")
    
    # Calculate relative path
    source_path = Path(source_file_path).resolve()
    header_path = Path(header_file_path).resolve()
    
    try:
        relative_path = header_path.relative_to(source_path.parent)
        include_line = f'#include "{relative_path}"\n'
        print(f"    Include line: {include_line.strip()}")
    except ValueError:
        include_line = f'#include "{header_path}"\n'
        print(f"    Include line (absolute): {include_line.strip()}")
    
    # Check if the same include already exists
    header_name = header_path.name
    for i, line in enumerate(lines):
        if header_name in line:
            print(f"    SKIP: Already exists at line {i+1}: {line.strip()}")
            return lines, 0
    
    # Determine insertion position
    insert_idx = first_use_line - 1
    
    # Safety check
    if insert_idx < 0:
        insert_idx = 0
        print(f"    Adjusted insert_idx to 0 (was negative)")
    elif insert_idx > len(lines):
        insert_idx = len(lines)
        print(f"    Adjusted insert_idx to {len(lines)} (was out of range)")
    
    print(f"    Initial insert position: line {insert_idx + 1}")
    
    # Insert after the existing #include block
    last_include_before_use = -1
    for i in range(min(insert_idx, len(lines))):
        if re.match(r'^\s*#\s*include\s+', lines[i]):
            last_include_before_use = i
            print(f"    Found existing #include at line {i+1}: {lines[i].strip()}")
    
    # If there is a #include block, insert right after it
    if last_include_before_use >= 0:
        insert_idx = last_include_before_use + 1
        print(f"    Adjusted to insert after last #include: line {insert_idx + 1}")
        # Skip empty lines
        while insert_idx < len(lines) and lines[insert_idx].strip() == '':
            insert_idx += 1
            print(f"    Skipped empty line, now at line {insert_idx + 1}")
    
    # Insert
    print(f"    INSERTING at line {insert_idx + 1}")
    lines.insert(insert_idx, include_line)
    
    return lines, 1



def check_is_in_target_dir(file_path, target_dir, program_files):

    if '<command line>' in str(file_path):
        return False 
    
    if '<built-in>' in str(file_path):
        return False 

    # if str(target_dir) in str(file_path):
    #     return True

    if not is_system_file(file_path, program_files):
        return True
    # try:
    #     abs_file = os.path.abspath(file_path)
    #     abs_target = os.path.abspath(target_dir)
    #     return abs_file.startswith(abs_target)
    # except:
    
    return False


def insert_include_before_line(lines, target_line, source_file_path, header_file_path):
    """
    Insert #include just before the specified line
    
    Args:
        lines: All lines of the file
        target_line: Line number to insert at (1-indexed)
        source_file_path: Path of the source file to add the include to
        header_file_path: Path of the header file to include
    
    Returns:
        (modified_lines, line_offset): Modified line list and number of lines added
    """
    # Calculate relative path
    source_path = Path(source_file_path).resolve()
    header_path = Path(header_file_path).resolve()
    
    try:
        relative_path = header_path.relative_to(source_path.parent)
        include_line = f'#include "{relative_path}"\n'
    except ValueError:
        include_line = f'#include "{header_path}"\n'
    
    # Check if the same include already exists
    header_name = header_path.name
    for line in lines:
        if header_name in line:
            return lines, 0
    
    # Determine insertion position (convert from 1-indexed to 0-indexed)
    insert_idx = max(0, target_line - 1)
    
    # Safety check
    if insert_idx > len(lines):
        insert_idx = len(lines)
    
    # Insert the include
    lines.insert(insert_idx, include_line)
    
    return lines, 1



def detect_include_guards(all_directive_path, target_dir, meta_dir, guards_path, is_program_path):
    """Detect include guards and save to JSON"""
    print("Detecting include guards...")
    
    include_guards = []
    
    data = read_json(all_directive_path)
    program_files = set(read_json(is_program_path))

    # Process per file
    for file_path, file_data in data.items():
        # Check if the file is within target_dir
        """
        if target_dir not in file_path:
            continue
        """
        
        if is_system_file(file_path, program_files):
            continue
        
        ifndef_list = file_data.get('ifndef', [])
        define_list = file_data.get('defined', [])
        
        if not ifndef_list or not define_list:
            continue
        
        # Index defines by macro name
        defines_by_name = {}
        for d in define_list:
            name = d['name']
            if name:
                if name not in defines_by_name:
                    defines_by_name[name] = []
                defines_by_name[name].append(d)
        
        # Check each ifndef
        for ifndef_info in ifndef_list:
            macro_name = ifndef_info['name']
            ifndef_line = ifndef_info.get('start_line')
            block_start = ifndef_info.get('block_start')
            block_end = ifndef_info.get('block_end')
            endif_info = ifndef_info.get('endif', {})
            endif_line = endif_info.get('start_line')
            
            if not macro_name or not ifndef_line or not endif_line:
                continue
            
            # Is there a define with the same name immediately after (within 1-2 lines)?
            matching_defines = defines_by_name.get(macro_name, [])
            define_found = None
            for d in matching_defines:
                def_line = d.get('start_line')
                if def_line and abs(def_line - ifndef_line) <= 2:
                    define_found = d
                    break
            
            if not define_found:
                continue
            
            def_line = define_found.get('start_line')
            guard_start = block_start or ifndef_line
            guard_end = block_end or endif_line
            
            # Check with metadata whether it encloses all other elements
            metadata, _ = obtain_metadata(file_path, meta_dir, False, None, "def")
            
            encloses_all = True
            if metadata:
                for key, meta in metadata.items():
                    meta_start = meta.get('block_start')
                    meta_end = meta.get('block_end')
                    
                    if meta_start is None or meta_end is None:
                        continue
                    
                    # Skip itself
                    if meta_start == guard_start and meta_end == guard_end:
                        continue
                    
                    # If out of range, it is not an include guard
                    if meta_start < guard_start or meta_end > guard_end:
                        encloses_all = False
                        break
            
            if not encloses_all:
                continue
            
            include_guards.append({
                'macro_name': macro_name,
                'file_path': file_path,
                'ifndef_line': ifndef_line,
                'define_line': def_line,
                'endif_line': endif_line
            })
            
            #print(f"  Found: {macro_name} in {file_path}")
    
    output_data = {
        'total_include_guards': len(include_guards),
        'guards': include_guards
    }
    
    write_json(guards_path, output_data)
    
    print(f"\nSaved {len(include_guards)} include guards to: {guards_path}\n")
    
    return include_guards




def detect_guarded_macros(all_directive_path, target_dir, guarded_macros_path, is_program_path):
    """Detect guarded macros and save to JSON
    
    A definition of M is guarded if its #define appears inside a #ifndef M block.
    Each definition is uniquely identified by macro_key (name:file:line:col).
    """
    print("Detecting guarded macros...")
    
    data = read_json(all_directive_path)
    
    # 1. Collect #ifndef blocks from all files
    #    key: (file_path, macro_name) -> list of (block_start, block_end)
    ifndef_blocks = {}
    for file_path, file_data in data.items():
        for ifndef_info in file_data.get('ifndef', []):
            macro_key = ifndef_info.get('macro_key')
            macro_name = ifndef_info['name']
            block_start = ifndef_info.get('block_start')
            block_end = ifndef_info.get('block_end')
            
            if not macro_name or block_start is None or block_end is None:
                continue
            ifndef_blocks.setdefault((file_path, macro_name), []).append({  #((file_path, macro_name), []).append({
                'block_start': block_start,
                'block_end': block_end,
                'ifndef_line': ifndef_info.get('start_line'),
            })
    
    # 2. Scan #define from all files and determine whether each is guarded
    guarded_macros = []
    seen = set()
    program_files = set(read_json(is_program_path))

    for file_path, file_data in data.items():
        # if target_dir not in file_path:
        #     continue
        
        if is_system_file(file_path, program_files):
            continue
        
        for d in file_data.get('defined', []):
            macro_name = d['name']
            macro_key = d.get('macro_key')
            def_line = d.get('start_line')
            if not macro_name or not macro_key or def_line is None:
                continue
            
            if macro_key in seen:
                continue

            # Whether there is a #ifndef block with the same name in the same file,
            # and this #define is contained within its range
            for block in ifndef_blocks.get((file_path, macro_name), []):
                if block['block_start'] <= def_line <= block['block_end']:
                    guarded_macros.append({
                        'macro_name': macro_name,
                        'macro_key': macro_key,
                        'file_path': file_path,
                        'define_line': def_line,
                        'ifndef_line': block['ifndef_line'],
                        'block_start': block['block_start'],
                        'block_end': block['block_end'],
                    })
                    
                    seen.add(macro_key)

                    break  # Finding one for this definition is sufficient
    
    output_data = {
        'total_guarded_macros': len(guarded_macros),
        'guards': guarded_macros
    }
    
    write_json(guarded_macros_path, output_data)
    
    print(f"\nSaved {len(guarded_macros)} guarded macros to: {guarded_macros_path}\n")
    
    return guarded_macros


def insert_guarded_flag(guarded_macros_path, gen_macro_usage_meta_path):
    print("Insert guarded flags...")

    guards = read_json(guarded_macros_path)
    usage_macros = read_json(gen_macro_usage_meta_path)

    # Set of guarded ifndef lines (for is_guard)
    guard_keys = set()
    # Set of guarded define definition locations (for is_guarded)
    guarded_define_keys = set()
    # added
    guard_to_guarded = {}
    guarded_to_guard = {}
    # ended
    for item in guards['guards']:
        file_path = item['file_path']
        ifndef_line = item['ifndef_line']
        define_line = item['define_line']

        guard_keys.add(f"{file_path}:{ifndef_line}")
        guarded_define_keys.add(f"{file_path}:{define_line}")
        # added
        guard_key = f"{file_path}:{ifndef_line}"
        define_key = f"{file_path}:{define_line}"
        guard_to_guarded[guard_key] = define_key
        guarded_to_guard[define_key] = guard_key
        # ended

    # Insert flags into each macro in usage_macros
    for macro in usage_macros['macros']:
        # is_guarded: this #define is inside a #ifndef block
        def_key = f"{macro['file_path']}:{macro['start_line']}"
        if def_key in guarded_define_keys:
            macro['is_guarded'] = True
            # added
            macro['guarded_by'] = guarded_to_guard[def_key]
            # ended

        # is_guard: this macro is used as a condition in #ifndef
        appearances = macro['appearances']
        for app in appearances:
            app_without_column = app.rsplit(":", 1)[0]
            if app_without_column in guard_keys:
                macro['is_guard'] = True
                # added
                macro['guarded'] = guard_to_guarded[app_without_column]
                # ended
                break

    write_json(gen_macro_usage_meta_path, usage_macros)


def read_file_lines(file_path, start, end):
    """Read lines in the specified range"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return ''.join(lines[start-1:end])


def get_guard_context(guard_info, context_lines=5):
    """Get the code range needed for include guard determination"""
    file_path = guard_info['file_path']
    ifndef_line = guard_info['ifndef_line']
    define_line = guard_info['define_line']
    endif_line = guard_info['endif_line']
    
    # Get total number of lines in the file
    with open(file_path, 'r') as f:
        total_lines = len(f.readlines())
    
    # Ranges needed for determination:
    # - context_lines lines before and after #ifndef
    # - context_lines lines right after #define (start of actual content)
    # - context_lines lines just before #endif (end of actual content)
    
    start_line = max(1, ifndef_line - context_lines)
    end_line = min(endif_line + context_lines, total_lines)
    
    # However, if the file is small, include the entire file
    if end_line - start_line < 30:
        start_line = 1
        end_line = total_lines

    return read_file_lines(file_path, start_line, end_line)



def divide_guards(guards_path, count):
    all_parts = []
    print("Dividing macro guards ...")

    with open(guards_path, 'r') as f:
        data = json.load(f)
    
    guards = data.get('guards', [])
    total = len(guards)
    
    if total == 0:
        return []

    for item in guards:
        item['code'] = get_guard_context(item)
    
    # Split into count number of parts
    chunk_size = (total + count - 1) // count  # Ceiling division
    
    for i in range(count):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        # Fix: do not add empty lists
        if start < total:  # Only add if start is less than total
            part = guards[start:end]
            if part:  # Just in case, confirm it is not empty
                all_parts.append(part)
    
    print(f"Divided {total} guards into {len(all_parts)} parts:")
    for i, part in enumerate(all_parts):
        print(f"  Part {i}: {len(part)} guards")

    return all_parts


def delete_guards(guards_path, target_dir, meta_dir, is_program_path):
    output_dir = "test_out"
    
    print("Deleting include guards...")
    
    with open(guards_path, 'r') as f:
        guards_data = json.load(f)
    
    guard_names = []
    for guard in guards_data.get('guards', []):
        guard_names.append(guard['macro_name'])

    # Collect lines to delete per file
    files_to_update = {}
    
    # Process items with is_include_guard=True from guards_data['guards']
    for guard in guards_data.get('guards', []):
        is_include_guard = guard.get('is_include_guard', False)
        if is_include_guard is not True:
            continue
        
        file_path = guard['file_path']
        if file_path not in files_to_update:
            files_to_update[file_path] = set()
        
        # Add the 3 lines (ifndef, define, endif) to deletion targets
        files_to_update[file_path].add(guard['ifndef_line'])
        files_to_update[file_path].add(guard['define_line'])
        files_to_update[file_path].add(guard['endif_line'])
        
        #print(f"  Deleting guard '{guard['macro_name']}' from {file_path}")
        #print(f"    Lines: ifndef={guard['ifndef_line']}, define={guard['define_line']}, endif={guard['endif_line']}")
    
    # Update each file
    program_files = set(read_json(is_program_path))

    for file_path, lines_to_delete in files_to_update.items():
        # if target_dir not in file_path:
        #     continue
        if is_system_file(file_path, program_files):
            continue

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Replace lines to delete with // deleted (line numbers are 1-based)
        for line_num in sorted(lines_to_delete):
            if 0 < line_num <= len(lines):
                lines[line_num - 1] = '// deleted\n'
        
        # Generate output file path
        output_path = file_path
        # rel_path = os.path.relpath(file_path, target_dir)
        # output_path = os.path.join(output_dir, rel_path)
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write back to file
        if file_path.startswith(os.path.abspath(target_dir)):# Check once more just to be safe
            with open(output_path, 'w') as f:
                f.writelines(lines)
            
        print(f"  Updated: {file_path} -> {output_path} ({len(lines_to_delete)} lines deleted)")
    
    print("Include guard deletion completed.")

    ## To also delete from metadata
    meta_files = get_all_files(meta_dir)
    for meta_path in meta_files:
        meta_data = read_json(meta_path)

        flattened = {}
        for comp_key, item in meta_data.items():
            if 'name' in item and item['name'] in guard_names:
                # If components exist, add their contents one level up
                if 'components' in item and item['components']:
                    for comp_key, comp_value in item['components'].items():
                        flattened[comp_key] = comp_value

            if 'macro_name' in item and item['macro_name'] in guard_names:
                # If components exist, add their contents one level up
                if 'components' in item and item['components']:
                    for comp_key, comp_value in item['components'].items():
                        flattened[comp_key] = comp_value

            else:
                flattened[comp_key] = item
        
        write_json(meta_path, flattened)


include_format = f"""
{{
    "answer" : [
        {{
            "macro_name" : (macro name),
            "file_path" : (file path),
            "ifndef_line" : (line of the macro ifndef),
            "endif_line" : (line of the macro endif),
            "is_include_guard" : True if the macro roles as a include guard, otherwise False,
        }},
        {{
            "macro_name" : (macro name),
            "file_path" : (file path),
            "ifndef_line" : (line of the macro ifndef),
            "endif_line" : (line of the macro endif),
            "is_include_guard" : True if the macro roles as a include guard, otherwise False,
        }},...
    ],
    "ongoing" : true if the response will continue. false otherwise,
    "reason" : explanatory text for the response (insert here if needed)
}}
"""

def check_guards_with_llms(llm_on, meta_dir, target_dir, guards_path, llm_answer_path, database_dir, 
                           llm_choice, llm_instance, token_path, chat_dir
                           ):

    LLM_GUARD = llm_on #False #True  #LLM_GUARD = False #True
    if LLM_GUARD:
        prompt = [f"We have picked up macro variables that are candidates for include guards satisfying the following conditions. Please determine whether each is a real include guard macro and respond.",
                    "## Conditions already satisfied:"
                    "  - #ifndef MACRO exists,",
                    "  - followed by #define MACRO",
                    "  - corresponding #endif exists",
                    "  - only 1 use site: len(uses) == 1 (include guards are only used in #ifndef)",
                    ""
                ]

        prompt.extend(["Please format your response as follows:"])
        prompt.extend([include_format])

        all_parts = divide_guards(guards_path, 10)

        for part in all_parts:
            prompt.extend(["# Macro variables:"])
            prompt.extend(part)

            ongoing_flag = None

            while(1):
                if ongoing_flag is False:
                    break
                    
                rsp_json = ask_llm(prompt, "continue", llm_instance)

                if 'ongoing' in rsp_json:
                    ongoing_flag = rsp_json['ongoing']
                
                if 'answer' in rsp_json:
                    answer_json = rsp_json['answer']
                    append_json(llm_answer_path, answer_json)


    print("=========== End of macro analysis ===========")

    # Load LLM answers
    with open(llm_answer_path, 'r') as f:
        llm_answers = json.load(f)
    
    # Extract macros determined to be include guards (strictly identified by file path, macro name, and ifndef line)
    true_guards = []
    for answer in llm_answers:
        if answer.get('is_include_guard') == True:
            macro_name = answer.get('macro_name')
            file_path = answer.get('file_path')
            ifndef_line = answer.get('ifndef_line')
            
            if macro_name and file_path and ifndef_line:
                true_guards.append({
                    'macro_name': macro_name,
                    'file_path': file_path,
                    'ifndef_line': ifndef_line
                })
    
    print(f"Found {len(true_guards)} include guards to delete")
    
    # Load LLM answers
    with open(llm_answer_path, 'r') as f:
        llm_answers = json.load(f)
    
    # Extract macros determined to be include guards (strictly identified by file path, macro name, and ifndef line)
    true_guards = []
    for answer in llm_answers:
        if answer.get('is_include_guard') == True:
            macro_name = answer.get('macro_name')
            file_path = answer.get('file_path')
            ifndef_line = answer.get('ifndef_line')
            
            if macro_name and file_path and ifndef_line:
                true_guards.append({
                    'macro_name': macro_name,
                    'file_path': file_path,
                    'ifndef_line': ifndef_line
                })
    
    print(f"Found {len(true_guards)} include guards to delete")
    
    # Load the guards file
    with open(guards_path, 'r') as f:
        guards_data = json.load(f)
    
    # Collect lines to delete per file
    files_to_update = {}
    
    for llm_guard in true_guards:
        # Find an exactly matching guard from guards_path
        matching_guard = None
        for guard in guards_data.get('guards', []):
            if (guard['file_path'] == llm_guard['file_path'] and
                guard['macro_name'] == llm_guard['macro_name'] and
                guard['ifndef_line'] == llm_guard['ifndef_line']):
                matching_guard = guard
                break
        
        # Fix: move indentation outside
        if matching_guard:
            matching_guard['is_include_guard'] = True
    
    # Fix: write back to the guards file
    with open(guards_path, 'w') as f:
        json.dump(guards_data, f, indent=4, ensure_ascii=False)
    
    print(f"Updated guards information in: {guards_path}")


# Not sure if this matches the independent path
def delete_independent_defs(independent_path, target_dir, is_program_path):
    """
    Load macro definition info from a JSON file, delete the corresponding lines,
    and insert // deleted comments.
    
    Args:
        independent_path: Path to the JSON file containing macro definition info
        target_dir: Directory containing the source files to process
    """

    print("delete_independent_defs...")
    print(independent_path)
    output_dir = "test_out"
 
    macro_data = read_json(independent_path)
    
    # Collect line ranges of macro definitions per file
    file_deletions = {}

    # if macro_data is None:
    #     return

    for item in macro_data:
        file_path = item.get('file_path')
        start_line = item.get('start_line')
        end_line = item.get('end_line')
        
        if not file_path or not start_line or not end_line:
            continue
        
        if file_path not in file_deletions:
            file_deletions[file_path] = []
        
        # Record all lines from start_line to end_line
        for line_num in range(start_line, end_line + 1):
            file_deletions[file_path].append(line_num)
    
    # Process each file
    program_files = set(read_json(is_program_path))
    for file_path, line_numbers in file_deletions.items():
        # if target_dir not in file_path:
        #     continue
        if is_system_file(file_path, program_files):
            continue

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Deduplicate and sort
        unique_lines = sorted(set(line_numbers))
        
        # Replace lines to delete with // deleted (line numbers are 1-based)
        for line_num in unique_lines:
            if 0 < line_num <= len(lines):
                lines[line_num - 1] = '// deleted\n'
        
        # Generate the output file path
        output_path = file_path
        # rel_path = os.path.relpath(file_path, target_dir)
        # output_path = os.path.join(output_dir, rel_path)
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write back to file
        if file_path.startswith(os.path.abspath(target_dir)):# Extra check just to be safe
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
        print(f"Processed: {file_path} -> {output_path} ({len(unique_lines)} lines deleted)")


def delete_global_defs(target_dir, meta_dir, is_program_path):

    output_dir = "test_out"
    file_deletions = {}
    global_vars = []

    meta_files = get_all_files(meta_dir)
    for meta_path in meta_files:
        meta_data = read_json(meta_path)
        #print(meta_path)

        for nema_key, item in meta_data.items():
            if 'kind' not in item:
                continue

            if item['kind'] == 'global_var':
                global_vars.append(item)
            
            if 'components' in item:
                for each_key, each_item in item['components'].items():
                    if 'kind' not in each_item:
                        continue
                    if each_item['kind'] == 'global_var':
                        global_vars.append(each_item)

    for item in global_vars:
        file_path = item.get('file_path')
        start_line = item.get('start_line')
        end_line = item.get('end_line')
        
        if not file_path or not start_line or not end_line:
            continue
        
        if file_path not in file_deletions:
            file_deletions[file_path] = []
        
        # Record all lines from start_line to end_line
        for line_num in range(start_line, end_line + 1):
            file_deletions[file_path].append(line_num)
    
    # Process each file
    program_files = set(read_json(is_program_path))
    for file_path, line_numbers in file_deletions.items():
        # if target_dir not in file_path:
        #     continue
        if is_system_file(file_path, program_files):
            continue

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Deduplicate and sort
        unique_lines = sorted(set(line_numbers))
        
        # Replace lines to delete with // deleted (line numbers are 1-based)
        for line_num in unique_lines:
            if 0 < line_num <= len(lines):
                lines[line_num - 1] = '// deleted\n'
        
        # Generate the output file path
        # rel_path = os.path.relpath(file_path, target_dir)
        # output_path = os.path.join(output_dir, rel_path)
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        

        # Write back to file
        if file_path.startswith(os.path.abspath(target_dir)):# Extra check just to be safe
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        
        print(f"Processed: {file_path} -> {output_path} ({len(unique_lines)} lines deleted)")



#def delete_conditional_macro_defs(all_directive_path, target_dir):  # I feel like this will inevitably introduce some distortion, but is it okay?
def delete_macro_defs(all_directive_path, target_dir, is_program_path):  # I feel like this will inevitably introduce some distortion, but is it okay?
    print(all_directive_path)
    output_dir = "test_out"

    """
    Load macro definition info from a JSON file, delete the corresponding lines,
    and insert // deleted comments.
    
    Args:
        all_directive_path: Path to the JSON file containing macro definition info
        target_dir: Directory containing the source files to process
        output_dir: Directory to output the processed files
    """
    print("delete_macro_defs")
    
    # Load the JSON file
    with open(all_directive_path, 'r', encoding='utf-8') as f:
        macro_data = json.load(f)
    
    # Collect line numbers of macro definitions per file
    file_deletions = {}
    
    for file_path, file_info in macro_data.items():
        if 'defined' in file_info:
            for macro_def in file_info['defined']:
                if macro_def['type'] == 'DEFINED':
                    real_file = macro_def['file_path']
                    line_num = macro_def['line']
                    
                    if real_file not in file_deletions:
                        file_deletions[real_file] = set()
                    file_deletions[real_file].add(line_num)
    
    # Process each file
    program_files = set(read_json(is_program_path))
    for file_path, line_numbers in file_deletions.items():
        # if target_dir not in file_path:
        #     continue
        if is_system_file(file_path, program_files):
            continue

        if not os.path.exists(file_path):
            continue
        # # Skip built-in or system files
        # if '<built-in>' in file_path or not os.path.exists(file_path):
        #     continue
        
        # # Check if the file is within target_dir
        # try:
        #     rel_path = os.path.relpath(file_path, target_dir)
        #     if rel_path.startswith('..'):
        #         continue
        # except ValueError:
        #     continue
        

        # print(f"Stop!!!!!: {all_directive_path}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Replace lines to delete with // deleted (line numbers are 1-based)
        for line_num in sorted(line_numbers):
            if 0 < line_num <= len(lines):
                lines[line_num - 1] = '// deleted\n'
        
        # Generate the output file path
        rel_path = os.path.relpath(file_path, target_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the file
        #"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        #"""
        
        print(f"Processed: {file_path} -> {output_path} ({len(line_numbers)} lines deleted)")



def convert_use_entry(use, parent_macro_key):

    if not use:
        return None
    
    use_name = use.get("name")
    use_def = use.get("definition")
    use_start_line = use.get("start_line")
    use_end_line = use.get("end_line")
    
    if not use_name:
        return None
    
    # Infer file_path from definition
    if use_def:
        use_file_path, _, _ = parse_def_loc(use_def)
    else:
        use_file_path = None
    
    use_macro_key = f"{use_name}:{use_def}" if use_def else f"{use_name}:undefined"
    
    # Convert to taken_macros format
    return {
        "type": "USE",  # gen_meta has no type info, so this is a placeholder value
        "file_path": use_file_path,
        "start_line": use_start_line,
        "start_column": None,  # gen_meta has no column info
        "end_line": use_end_line,
        "end_column": None,
        "macro_key": use_macro_key
    }


def convert_gen_meta(gen_meta_path):
    taken_macros = {}
    gen_meta = read_json(gen_meta_path)

    for entry in gen_meta.get("symbols", []):
        uses = entry.get("uses", [])
        for item in uses:
            if 'macro' not in item .get("kind"):
                continue

            name = item['name']
            macro_key = f"{name}:{item['definition']}"

            # Convert to taken_macros format
            if macro_key not in taken_macros:
                taken_macros[macro_key] = {}
            
            if 'definition' not in taken_macros[macro_key]:
                taken_macros[macro_key] = {
                    "name": macro_key.split(":")[0],
                    "definition": parse_definition_from_key(macro_key),
                }

            if 'uses' not in taken_macros[macro_key]:
                taken_macros[macro_key]['uses'] = []

            # if 'start_line' not in item:
            #     print(item)
            
            use_file_path, use_line, use_colum = parse_def_loc(item['usage_location'])
            taken_macros[macro_key]['uses'].append({
                "file_path" : use_file_path, #entry['file_path'],  # Using entry here seems to be a key detail
                "start_line" : use_line, #item['start_line'],
            })

    return taken_macros



def convert_macro_usage(gen_meta_path):
    taken_macros = {}
    gen_meta = read_json(gen_meta_path)

    for entry in gen_meta.get("macros", []):
        name = entry['name']
        uses = entry.get("appearances", [])
        definition = entry['definition']
        is_const = entry['is_const']
        #is_independent = entry['is_independent']
        is_flag = entry['is_flag']

        if 'is_independent' not in entry:
            entry['is_independent'] = False
        is_independent = entry['is_independent']

        if 'is_guard' not in entry:
            entry['is_guard'] = False
        is_guard = entry['is_guard']

        if 'is_guarded' not in entry:
            entry['is_guarded'] = False
        is_guarded = entry['is_guarded']

        if 'guarded' not in entry:
            entry['guarded'] = False
        guarded = entry['guarded']

        macro_key = f"{name}:{definition}"

        # ★ Create the entry even if appearances is empty
        if macro_key not in taken_macros:
            taken_macros[macro_key] = {
                "name": name,
                "definition": parse_definition_from_key(macro_key),
                "is_const": is_const,
                "is_independent" : is_independent,
                "is_flag": is_flag,
                "is_guard": is_guard,
                "is_guarded": is_guarded,
                "guarded" : guarded,
                "appearances": []
            }

        for item in uses:
            # Convert to taken_macros format
            if macro_key not in taken_macros:
                taken_macros[macro_key] = {}
            
            if 'definition' not in taken_macros[macro_key]:
                taken_macros[macro_key] = {
                    "name": macro_key.split(":")[0],
                    "definition": parse_definition_from_key(macro_key),
                }

            if 'appearances' not in taken_macros[macro_key]:
                taken_macros[macro_key]['appearances'] = []

            # if 'start_line' not in item:
            #     print(item)
            
            use_file_path, use_line, use_colum = parse_def_loc(item)
            taken_macros[macro_key]['appearances'].append({
                "file_path" : use_file_path, #entry['file_path'],  # Using entry here seems to be a key detail
                "start_line" : use_line, #item['start_line'],
            })

    return taken_macros 
    

def convert_gen_macro_meta(gen_macro_meta_path):

    print(f"Converting {gen_macro_meta_path} to taken_macros format...")
    gen_macro_meta = read_json(gen_macro_meta_path)
    
    taken_macros = {}
    
    # Process macro info from gen_macro_meta.json
    for macro_entry in gen_macro_meta.get("macros", []):
        if 'macro' not in macro_entry.get("kind"): # != "macro_object":
            continue
        
        macro_name = macro_entry.get("name")
        definition_str = macro_entry.get("definition")  # "file:line:column"
        uses = macro_entry.get("uses", [])
        
        if not macro_name or not definition_str:
            continue
        
        # Parse the definition string
        def_file_path, def_line, def_col = parse_def_loc(definition_str)
        
        # Generate macro_key
        macro_key = f"{macro_name}:{definition_str}"
        
        # Convert uses
        for use in uses:
            use_entry = convert_use_entry(use, macro_key)

            # Convert to taken_macros format
            if macro_key not in taken_macros:
                taken_macros[macro_key] = []

            if 'definition' not in taken_macros[macro_key]:
                taken_macros[macro_key] = {
                    "name": macro_key.split(":")[0],
                    "definition": parse_definition_from_key(macro_key),
                }

            if 'uses' not in taken_macros[macro_key]:
                taken_macros[macro_key]['uses'] = []

            taken_macros[macro_key]['uses'].append({
                "file_path" : entry['file_path'],  # Using entry here seems to be a key detail
                "start_line" : item['start_line'],
            })
    
    return taken_macros



def is_duplicate_use(new_use, existing_uses):
    for existing in existing_uses:
        if (existing.get("file_path") == new_use.get("file_path") and
            existing.get("start_line") == new_use.get("start_line") and
            existing.get("start_column") == new_use.get("start_column", existing.get("start_column"))):
            return True
    return False


def parse_definition_from_key(macro_key):
    parts = macro_key.split(":")

    #print(parts)
    if len(parts) >= 4:
        file_path = ":".join(parts[1:-2]) 
        line = int(parts[-2])
        column = int(parts[-1])
        return {
            "file_path": file_path,
            "start_line": line,
            "start_column": column
        }
    return {}


def is_directive(use_item, macro):
    file_path = use_item['file_path']
    start_line = use_item['start_line']

    for item in macro['appearances']:
        if item['file_path'] == file_path and item['start_line'] == start_line:
            if 'type' in item and item['type'] in ["IFDEF", "IFNDEF", "IF", "ELIF"]:
                return True
    return False



def get_taken_macros(taken_directive_path, gen_macro_usage_meta_path, taken_macros_path, database_dir):  #  gen_meta_path,   # , gen_macro_meta_path

    # meta_data = convert_gen_meta(gen_meta_path) # Since macros are now separated, this may no longer be needed?
    meta_data = {}

    #macro_meta_data = convert_gen_macro_meta(gen_macro_meta_path)
    macro_meta_data = {}

    macro_usage_meta_data = convert_macro_usage(gen_macro_usage_meta_path) #, meta_dir)


    # Merge taken_directive_path, meta_data, and macro_meta_data, then save to taken_macros_path
    if os.path.exists(taken_directive_path):
        taken_macros = read_json(taken_directive_path)
    else:
        taken_macros = {}

    write_json(f"{database_dir}/tmp.json", meta_data)
    write_json(f"{database_dir}/tmp2.json", macro_usage_meta_data)

    for macro_key, use_item in meta_data.items():
        new_uses = use_item['appearances']

        if macro_key not in taken_macros: # Is this needed?
            taken_macros[macro_key] = {}
            taken_macros[macro_key]["appearances"] = []

        # Add uses without duplicates
        existing_uses = taken_macros[macro_key]["appearances"]
        for new_use in new_uses:
            #new_entry = parse_definition_from_key(new_use)
            # new_entry = {
            #     "file_path": file_path,
            #     "start_line": line,
            #     "start_column": column
            # }
            if not is_duplicate_use(new_use, existing_uses):
                taken_macros[macro_key]["appearances"].append(new_use)


    for macro_key, macro_item in macro_usage_meta_data.items():
        new_uses = macro_item['appearances']

        if macro_key not in taken_macros: # Is this needed?
            if "unknown" in macro_key:
                notation = "unknown"
            elif "external" in macro_key:
                notation = "external"
            else:
                raise ValueError(f"Something wrong at {macro_key}")
            
            # taken_macros[macro_key] = dict(macro_item)
            # if "appearances" not in taken_macros[macro_key]:
            #     taken_macros[macro_key]["appearances"] = []
                
            #taken_macros[macro_key] = {}
            #taken_macros[macro_key]["appearances"] = []
            #"""
            #taken_macros[macro_key]["appearances"] = []
            #app_file, app_line, app_col = parse_def_loc(macro_item.get('definition'))
            taken_macros[macro_key] = {
                "name": macro_item.get('name'),
                "definition": {
                    "file_path": notation, #app_file,
                    "start_line": notation, #app_line,
                    "start_column": notation, #app_col,
                    "end_line": notation, #macro_item['end_line'],
                    "end_column": None,
                    "active": True,
                    "skipped": False,
                    "body": None
                },
                "signature": None, #macro_item.get('signature'),
                "appearances": []
            }
            #"""

        taken_macros[macro_key]['is_const'] = macro_item['is_const']
        taken_macros[macro_key]['is_independent'] = macro_item['is_independent']
        taken_macros[macro_key]['is_flag'] = macro_item['is_flag']
        taken_macros[macro_key]['is_guard'] = macro_item['is_guard']
        taken_macros[macro_key]['is_guarded'] = macro_item['is_guarded']
        taken_macros[macro_key]['guarded'] = macro_item['guarded']

        # Add uses without duplicates
        existing_uses = taken_macros[macro_key]["appearances"]
        for new_use in new_uses:
            if not is_duplicate_use(new_use, existing_uses):
                taken_macros[macro_key]["appearances"].append(new_use)

    """
    # Integrate macro_meta_data in the same manner
    for macro_key, new_uses in macro_meta_data.items():
        existing_uses = taken_macros[macro_key]["uses"]
        for new_use in new_uses:
            if not is_duplicate_use(new_use, existing_uses):
                existing_uses.append(new_use)
    """

    # Record whether each usage location is directive-based
    for macro_key, item in taken_macros.items():
        for use_item in item['appearances']:
            if is_directive(use_item, taken_macros[macro_key]):
                use_item['is_directive'] = True

    write_json(taken_macros_path, taken_macros)
    return taken_macros



def summarize_components(file_path, target_dir, meta_dir):
    """
    Move nested elements into components while preserving the hierarchical structure.
    Only non-nested elements remain at the top level.
    """

    meta_data, meta_path = obtain_metadata(file_path, meta_dir, False, None, "def")
    if meta_data is None:
        return
    #print(meta_path)
    
    # Initialize the components field for each item
    for key, item in meta_data.items():
        item['components'] = {}
    
    nested_item_keys = set()
    
    # ✅ Sort by start_line for faster processing
    items_list = sorted(meta_data.items(), key=lambda x: x[1].get('start_line', 0))
    
    # Analyze containment relationships
    for i, (key_i, item) in enumerate(items_list):
        start_line = item.get('start_line', 0)
        end_line = item.get('end_line', start_line)

        # ✅ Initialize block_end (starting from its own end_line)
        block_end = item.get('block_end', end_line)

        # if 'endif' in item:
        #     endif_info = item.get('endif', start_line)
        #     end_line = endif_info.get('start_line', start_line)
        
        # Manage uses as a dictionary
        uses_list = item.get('uses', [])
        uses_dict = {use.get('name'): use for use in uses_list if use.get('name')}
        
        # ✅ Since sorted by start_line, exit early if out of range
        for j in range(i + 1, len(items_list)):  # Start from i+1 (skip items before self)
            key_j, other_item = items_list[j]
            
            other_start_line = other_item.get('start_line', 0)
            other_end_line = other_item.get('end_line', other_start_line)

            # if 'endif' in other_item:
            #     endif_info = other_item.get('endif', start_line)
            #     end_line = endif_info.get('start_line', start_line)
                
            
            # ✅ Early exit: stop if other_item is outside the range of item
            if other_start_line > end_line:
                break
            
            # Containment check: item contains other_item
            if start_line <= other_start_line and other_end_line <= end_line:
                # Exclude exact matches
                if not (start_line == other_start_line and end_line == other_end_line):
                    if key_j not in item['components']:
                        # ✅ Copy the complete data (this is correct)
                        item['components'][key_j] = dict(other_item)
                        nested_item_keys.add(key_j)

                    # ✅ Update block_end (also considering block_end of nested elements)
                    other_block_end = other_item.get('block_end', other_end_line)
                    block_end = max(block_end, other_block_end)
                        
                    # Merge uses
                    other_uses_list = other_item.get('uses', [])
                    for other_use in other_uses_list:
                        other_use_name = other_use.get('name')
                        if other_use_name and other_use_name not in uses_dict:
                            uses_dict[other_use_name] = other_use
        
        
        # ✅ Set block_end after the loop ends
        item['block_end'] = block_end
        item['uses'] = list(uses_dict.values())  # Update uses
    
    # Remove nested elements from the top level
    top_level_items = {
        key: item 
        for key, item in meta_data.items() 
        if key not in nested_item_keys
    }

    # Add after the existing loop
    # ✅ Recalculate block_end (after components are finalized)
    for key, item in top_level_items.items():
        if item['components']:
            max_block_end = item.get('end_line', 0)
            for comp_key, comp in item['components'].items():
                comp_block_end = comp.get('block_end', comp.get('end_line', 0))
                max_block_end = max(max_block_end, comp_block_end)
            item['block_end'] = max_block_end
            
    
    # Write the updated metadata
    write_json(meta_path, top_level_items)
    
    # print(f"Total items: {len(meta_data)}")
    # print(f"Nested items: {len(nested_item_keys)}")
    # print(f"Top-level items: {len(top_level_items)}")
    
    return len(nested_item_keys)


# block_end didn't exist at this point, so it's not used here, but using it might simplify things
# The contents of (all_macros_path, outermost_path, guards_path, target_dir) — isn't it taken_macros_path instead of all_macros_path!? No, it's all
def get_outermost(all_directive_path, outermost_path, guards_path, target_dir, is_program_path):
    print(f"Getting outermost for {all_directive_path}")
    
    # Load the all_conds file
    all_conds = read_json(all_directive_path)
    
    # Load the guards file
    guards_data = read_json(guards_path)
    
    # Store include guard info in a set (for fast lookup)
    include_guards = set()
    for guard in guards_data.get('guards', []):
        # is_guard = guard.get('is_include_guard', False)
        # if is_guard:
        # Identify by tuple of (file_path, ifndef_line, endif_line)
        include_guards.add((
            guard['file_path'],
            guard['ifndef_line']
            #guard['endif_line']
        ))
    
    outermost_blocks = {}
    
    # Process per file
    program_files = set(read_json(is_program_path))

    for file_path, file_data in all_conds.items():
        # Only process files within target_dir
        # if target_dir not in file_path:
        #     continue
        if is_system_file(file_path, program_files):
            continue
        
        # Collect conditional directives for each file
        all_directives = []
        
        # Merge ifdef, ifndef, if, elif
        for directive_type in ['ifdef', 'ifndef', 'if', 'elif', 'else']: # , 'else' added
            for directive in file_data.get(directive_type, []):
                start_line = directive.get('start_line')

                endif_info = directive.get('endif', {})
                end_line = endif_info.get('start_line') # This is the key point
                
                if not (start_line and end_line):
                    continue
                
                # Check if it's an include guard
                directive_file = directive.get('file_path', file_path)
                #if (directive_file, start_line, end_line) in include_guards:
                if (directive_file, start_line) in include_guards:
                    #print(f"  Skipping include guard at {directive_file}:{start_line}")
                    continue
                
                all_directives.append({
                    'start': start_line,
                    'end': end_line,
                    'type': directive_type,
                    'data': directive
                })
        
        if not all_directives:
            continue
        
        # Sort by start_line
        all_directives.sort(key=lambda x: x['start'])
        outermost = []

        # Handle containment cases
        for directive in all_directives:
            merged = False
            
            for i, outer in enumerate(outermost):
                # If there is overlap or containment
                if (directive['start'] <= outer['end'] and 
                    directive['end'] >= outer['start']):
                    # Extend the range and merge
                    outermost[i] = {
                        'start': min(outer['start'], directive['start']),
                        'end': max(outer['end'], directive['end']),
                        'type': outer['type'], # Uses the info from the one processed first (outer)
                        'data': outer['data'] # Uses the info from the one processed first (outer)
                    }
                    merged = True
                    break
            if not merged:
                outermost.append(directive)
                
        # Store only data in the list
        file_result = [block['data'] for block in outermost]
        outermost_blocks[file_path] = file_result
    
    write_json(outermost_path, outermost_blocks)
    
    total_outermost = sum(len(blocks) for blocks in outermost_blocks.values())
    print(f"Found {total_outermost} outermost blocks (excluding include guards)")
    print(f"Saved to: {outermost_path}")
    
    return outermost_blocks



class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}
        self.rank = {k: 0 for k in keys}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def merge_overlapping_blocks(meta_data):
    """Merge intersecting blocks (the one with the smallest start line becomes the parent)"""
    if not meta_data:
        return meta_data
    
    keys = list(meta_data.keys())
    uf = UnionFind(keys)
    
    # Use IntervalTree for fast intersection detection
    #print(f"[DEBUG] merge_overlapping_blocks start: {len(meta_data)} items")

    tree = IntervalTree()
    for key, item in meta_data.items():
        start = item.get('block_start', 0)
        end = item.get('block_end', start)
        tree[start:end+1] = key
        #print(f"  {key}: [{start}, {end}]")
    
    # Union intersecting pairs
    for key, item in meta_data.items():
        start = item.get('block_start', 0)
        end = item.get('block_end', start)
        
        for interval in tree[start:end+1]:
            other_key = interval.data
            if other_key != key:
                uf.union(key, other_key)
    
    # Group by root
    groups = {}
    for key in keys:
        root = uf.find(key)
        if root not in groups:
            groups[root] = []
        groups[root].append(key)
    
    # Merge each group
    result = {}
    for root, group_keys in groups.items():
        if len(group_keys) == 1:
            # No merge needed
            result[group_keys[0]] = dict(meta_data[group_keys[0]])
        else:
            # The one with the smallest start line becomes the parent
            group_keys.sort(key=lambda k: meta_data[k].get('block_start', 0))
            parent_key = group_keys[0]
            child_keys = group_keys[1:]
            
            parent_item = dict(meta_data[parent_key])
            
            # Extend the range
            all_starts = [meta_data[k].get('block_start', 0) for k in group_keys]
            all_ends = [meta_data[k].get('block_end', meta_data[k].get('block_start', 0)) for k in group_keys]
            parent_item['block_start'] = min(all_starts)
            parent_item['block_end'] = max(all_ends)
            
            # Add children to components
            if 'components' not in parent_item:
                parent_item['components'] = {}
            
            for child_key in child_keys:
                parent_item['components'][child_key] = dict(meta_data[child_key])
            
            # Merge uses (deduplicated)
            uses_dict = {u.get('name'): u for u in parent_item.get('uses', []) if u.get('name')}
            for child_key in child_keys:
                for use in meta_data[child_key].get('uses', []):
                    name = use.get('name')
                    if name and name not in uses_dict:
                        uses_dict[name] = use
            parent_item['uses'] = list(uses_dict.values())
            
            result[parent_key] = parent_item
    
    return result



def combine_with_outermost_conditioned_blocks(all_directive_path, outermost_path, guards_path, target_dir, round_id, meta_dir, is_program_path): # file_path, 
    print(f"\n---- Combining with outermost conditioned blocks for {target_dir} ----")

    all_conds = get_outermost(all_directive_path, outermost_path, guards_path, target_dir, is_program_path)

    for file_path, file_conds in all_conds.items():
        meta_data, meta_path = obtain_metadata(file_path, meta_dir, False, None, "def")
        if meta_data is None:
            meta_data = {}

        meta_data.update(
            (f"{item['type']}:{item['file_path']}:{item['block_start']}", item)
            for item in file_conds
        )

        write_json(meta_path, meta_data)

    """
    #meta_data, meta_path = obtain_metadata(file_path, meta_dir, False, None, "def")
    meta_files = get_all_files(meta_dir)
    for meta_path in meta_files:
        file_path = get_file_path_from_meta_path(meta_path, meta_dir)

        meta_data = read_json(meta_path)

        if meta_data is None:
            #return
            meta_data = {} # Isn't this needed to add conditional blocks?
        
        # with open(outermost_path, 'r') as f:
        #     all_conds = json.load(f)
        #print(outermost_path)

        if file_path not in all_conds:
            continue
        file_conds = all_conds[file_path]  #all_conds.get(file_path, [])

        reformed_file_conds = {}
        for item in file_conds:
            tmp_key = f"{item['type']}:{item['file_path']}:{item['block_start']}" #item['start_line']}"
            reformed_file_conds[tmp_key] = item
        
        merged = meta_data | reformed_file_conds
        meta_data = merged

        write_json(meta_path, meta_data)
    """ 


def get_innermost(all_directive_path, innermost_path, target_dir, is_program_path):  # , guards_path
    print(f"Getting innermost for {all_directive_path}")
    
    all_conds = read_json(all_directive_path)

    """
    guards_data = read_json(guards_path)
    
    # Store include guard info in a set (for fast lookup)
    include_guards = set()
    for guard in guards_data.get('guards', []):
        # is_guard = guard.get('is_include_guard', False)
        # if is_guard:
        # Identify by tuple of (file_path, ifndef_line, endif_line)
        include_guards.add((
            guard['file_path'],
            guard['ifndef_line']
            #guard['endif_line']
        ))
    """
    
    innermost_blocks = {}
    program_files = set(read_json(is_program_path))


    for file_path, file_data in all_conds.items():
        # if target_dir not in file_path: # Only process files within target_dir
        #     continue
        if is_system_file(file_path, program_files):
            continue

        all_directives = [] # Collect conditional directives for each file
        
        # Merge ifdef, ifndef, if, elif
        for directive_type in ['ifdef', 'ifndef', 'if', 'elif', 'else']: # , 'else' added
            for directive in file_data.get(directive_type, []):
                start_line = directive.get('start_line')

                endif_info = directive.get('endif', {})
                end_line = endif_info.get('start_line') # This is the key point
                
                if not (start_line and end_line):
                    continue
                
                # Check if it's an include guard
                directive_file = directive.get('file_path', file_path)
                #if (directive_file, start_line, end_line) in include_guards:
                """
                if (directive_file, start_line) in include_guards:
                    #print(f"  Skipping include guard at {directive_file}:{start_line}")
                    continue
                """
                
                all_directives.append({
                    'start': start_line,
                    'end': end_line,
                    'type': directive_type,
                    'data': directive
                })
        
        if not all_directives:
            continue
        
        # Sort by start_line
        all_directives.sort(key=lambda x: x['start'])
        innermost = []
                
        # Store only data in the list
        file_result = [block['data'] for block in all_directives]
        innermost_blocks[file_path] = file_result
    

    write_json(innermost_path, innermost_blocks)  # Save to JSON

    total_innermost = sum(len(blocks) for blocks in innermost_blocks.values()) # Statistics
    print(f"Found {total_innermost} innermost blocks (excluding include guards)")
    print(f"Saved to: {innermost_path}")
    
    return innermost_blocks


def combine_with_innermost_conditioned_blocks(all_directive_path, target_dir, database_dir, round_id, div_meta_dir, is_program_path):  # , guards_path

    print(f"\n---- Combining with innermost conditioned blocks for {target_dir} ----")

    innermost_path = f"{database_dir}/innermost.json"
    all_conds = get_innermost(all_directive_path, innermost_path, target_dir, is_program_path)  # , guards_path

    for file_path, file_conds in all_conds.items():

        meta_data, meta_path = obtain_metadata(file_path, div_meta_dir, False, None, "def")
        if meta_data is None:
            meta_data = {}

        # Collect ranges of non-conditional blocks
        non_cond_ranges = []
        for key, meta in meta_data.items():
            is_cond = False
            if 'type' in meta:
                item_type = meta.get('type', '')
                if item_type in ['IF', 'IFDEF', 'IFNDEF', 'ELSE', 'ELIF']:
                    is_cond = True
            
            if is_cond is False:
                start = meta.get('block_start')
                end = meta.get('block_end')
                if start and end and start != end:
                    non_cond_ranges.append((start, end))
        
        # print(f"DEBUG: non_cond_ranges = {non_cond_ranges}")
        # print(f"DEBUG: meta_data keys = {list(meta_data.keys())}")

        ####
        for item in file_conds:
            block_start = item.get('block_start')
            block_end = item.get('block_end')

            # Is it fully enclosed within a function, etc.?
            is_enclosed = False
            if block_start and block_end:
                for nc_start, nc_end in non_cond_ranges:
                    if nc_start < block_start and block_end < nc_end:
                        is_enclosed = True
                        break

            if is_enclosed:
                # Remove from meta_data if it already exists
                key = f"{item['type']}:{item['file_path']}:{item['block_start']}"
                if key in meta_data:
                    del meta_data[key]
                continue  # Exclude conditional blocks inside functions

            key = f"{item['type']}:{item['file_path']}:{item['block_start']}"
            if key not in meta_data:
                meta_data[key] = item

        write_json(meta_path, meta_data)



def define_conditioned_blocks(file_path, target_dir, round_id, meta_dir):  # , outermost_path
    print(f"\n---- Define conditioned blocks for {file_path} ----")
    

    meta_data, meta_path = obtain_metadata(file_path, meta_dir, False, None, "def")
    if meta_data is None:
        #return
        meta_data = {} # Isn't this needed to add conditional blocks?
    
    """
    with open(outermost_path, 'r') as f:
        all_conds = json.load(f)
    
    print(outermost_path)
    if file_path not in all_conds:
        return
    file_conds = all_conds[file_path]  #all_conds.get(file_path, [])

    #file_conds = all_conds.get(file_path, [])

    # if not file_conds:
    #     write_json(meta_path, meta_data)
    #     return
    
    reformed_file_conds = {}
    for item in file_conds:
        tmp_key = f"{item['type']}:{item['file_path']}:{item['block_start']}" #item['start_line']}"
        reformed_file_conds[tmp_key] = item
    
    merged = meta_data | reformed_file_conds
    meta_data = merged
    """

    ###
    # ✅ Addition: Pre-merge intersecting blocks (smallest start line becomes parent)
    meta_data = merge_overlapping_blocks(meta_data)

    # Initialize the components field for each item
    for key, item in meta_data.items():
        if 'components' not in item:  # Skip if already merged
            item['components'] = {}
    ### 

    nested_item_keys = set()
    #"""
    # ✅ Build IntervalTree
    tree = IntervalTree()
    for key, item in meta_data.items():
        start_line = item.get('block_start', 0) #item.get('start_line', 0)
        end_line = item.get('block_end', start_line) #item.get('end_line', start_line)

        # Register as half-open interval [start, end+1) (IntervalTree specification)
        tree[start_line:end_line+1] = (key, item)
    
    # ✅ Fast containment search for each item
    for key, item in meta_data.items():
        start_line = item.get('block_start', 0)  # item.get('start_line', 0)
        end_line = item.get('block_end', start_line)  # item.get('end_line', start_line)

        # Search for items overlapping this range O(log n + k)
        overlapping = tree[start_line:end_line+1]
        
        uses_dict = {use.get('name'): use for use in item.get('uses', []) if use.get('name')}
        
        for interval in overlapping:
            other_key, other_item = interval.data
            
            if other_key == key:  # Skip self
                continue
            
            other_start_line = other_item.get('block_start', 0)
            other_end_line = other_item.get('block_end', other_start_line) #other_item.get('end_line', other_start_line)

            # Containment check
            if start_line <= other_start_line and other_end_line <= end_line:
                if not (start_line == other_start_line and end_line == other_end_line):
                    if other_key not in item['components']:
                        item['components'][other_key] = copy.deepcopy(other_item) #dict(other_item)
                        nested_item_keys.add(other_key)
                    
                    # Merge uses
                    for other_use in other_item.get('uses', []):
                        other_use_name = other_use.get('name')
                        if other_use_name and other_use_name not in uses_dict:
                            uses_dict[other_use_name] = other_use

        item['uses'] = list(uses_dict.values())
    #"""

    """
    # ✅ Fast containment search for each item
    for key, item in meta_data.items():
        start_line = item.get('block_start', 0)  # item.get('start_line', 0)
        end_line = item.get('block_end', start_line)  # item.get('end_line', start_line)

        uses_dict = {use.get('name'): use for use in item.get('uses', []) if use.get('name')}

        for other_key, other_item in meta_data.items():
            if other_key == key:
                continue
            
            other_start_line = other_item.get('block_start', 0)
            other_end_line = other_item.get('block_end', other_start_line) #other_item.get('end_line', other_start_line)


            # Containment check
            if start_line <= other_start_line and other_end_line <= end_line:
                if not (start_line == other_start_line and end_line == other_end_line):
                    if other_key not in item['components']:
                        item['components'][other_key] = copy.deepcopy(other_item) #dict(other_item)
                        nested_item_keys.add(other_key)
                    
                    # Merge uses
                    for other_use in other_item.get('uses', []):
                        other_use_name = other_use.get('name')
                        if other_use_name and other_use_name not in uses_dict:
                            uses_dict[other_use_name] = other_use

        item['uses'] = list(uses_dict.values())
    """

    # Remove nested elements from the top level
    top_level_items = {
        key: item 
        for key, item in meta_data.items() 
        if key not in nested_item_keys
    }
    
    # Sort by block_start
    sorted_top_level_items = dict(
        sorted(top_level_items.items(), key=lambda x: x[1].get('block_start', 0))
    )

    write_json(meta_path, sorted_top_level_items) #top_level_items)
    
    # print(f"Total items: {len(meta_data)}")
    # print(f"Nested items: {len(nested_item_keys)}")
    # print(f"Top-level items: {len(top_level_items)}")
    
    return len(nested_item_keys)


def define_blocks(round_id, all_directive_path, guards_path, target_dir, meta_dir, div_meta_dir, database_dir):  # , raw_dir

    # Is it okay to skip this? # Does it make sense to call summarize_components twice here?
    # p_f(summarize_components, target_dir, True, True, meta_dir)

    outermost_path = f"{database_dir}/outermost.json"
    #p_f(combine_with_outermost_conditioned_blocks, target_dir, True, True, round_id, all_conds, meta_dir)
    combine_with_outermost_conditioned_blocks(all_directive_path, outermost_path, guards_path, target_dir, round_id, meta_dir, is_program_path) # True, True, 

    if round_id == "3":
        # I think metadata should be stored separately for non-grouped and parallel ones.
        parent = os.path.dirname(div_meta_dir)
        copy_directory(meta_dir, parent)

    p_f(define_conditioned_blocks, target_dir, True, True, round_id, meta_dir) # # , all_macro_path dep_json_path # blocks.py, # analyze_blocks



def insert_macro_deps(macro_dep_path, target_dir, meta_dir):
    print("insert_macro_deps...")
    macro_symbols = read_json(macro_dep_path)

    for file_path, symbols in macro_symbols.items():
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping non-existent file: {file_path}")
            continue

        meta_path = obtain_metadata(file_path, meta_dir, False, True, "def")
        meta_path = Path(meta_path)

        if not os.path.exists(meta_path):
            existing_data = {}
        else:
            existing_data = read_json(meta_path)

        # Add a new symbol（key: "name:file_path:start_line"）
        for symbol in symbols:
            name = symbol['name']
            start_line = symbol['start_line']
            uses = symbol['uses']
            
            # Create key: "name:file_path:start_line"
            item_key = f"{name}:{file_path}:{start_line}"
            
            # Check duplicates（add if the key does not exist）
            if item_key not in existing_data:
                #existing_data[item_key] = symbol
                raise ValueError("Error in finding item_key")

            for use_item in uses:
                if 'macro' in use_item.get('kind'):
                    continue
                
                if 'uses' not in existing_data[item_key]:
                    existing_data[item_key]['uses'] = []
                existing_data[item_key]['uses'].append(use_item)
                
        write_json(meta_path, existing_data)
        #print(f"  ✅ {file_path}: {len(symbols)} symbols")



def insert_ifdef_statement(cfg_path, target_dir, meta_dir):
    print("Inserting if statement...")

    cfg_macros = read_json(cfg_path)
    all_cfs = {}

    if cfg_macros is None: # Need check. Is this true?
        return
    
    for item in cfg_macros:
        file_path = item['file_path']
        if file_path not in all_cfs:
            all_cfs[file_path] = []
        all_cfs[file_path].append(item)
    
    for file, data in all_cfs.items():
        file_path = os.path.abspath(file)
        # print(file_path)
        
        meta_data, meta_path = obtain_metadata(file_path, meta_dir, False, None, 'def')
        if meta_data is None:
            continue

        for item in data:
            name_key = f"IFDEF:{file_path}:{item['start_line']}"  # {item['macro_name']}
            #name_key = get_name_key(item)
            if name_key in meta_data:
                meta_data[name_key]['ifdef_statement'] = True

            # meta_data[name_key] = {
            #     "kind" : "ifdef_statement",
            #     "name": item['macro_name'],
            #     "definition": None,
            #     "file_path" : os.path.abspath(file_path), #file_path,
            #     "start_line": item['start_line'],
            #     "start_column": item['start_column'],
            #     "end_line": item['start_line'], #item['end_line'],  # need this?
            #     "end_column": item['start_column'], #None, # item['end_column'], # need this?
            #     "block_start": item['start_line'],
            #     "block_end": item['start_line'],
            #     "rust_code": {
            #         "file_path": None,
            #         "start_line": None
            #     },
            #     "uses": [],
            # }

        write_json(meta_path, meta_data)    


# Static configuration → Cargo.toml
# Compile-time dynamic processing → build.rs
# Runtime code → lib.rs / main.rs

"""
Condition                     build.rs   lib.rs
Determined at compile time       ✓         △
Changes at runtime               ✗         ✓
Hot path (frequently executed)   ✓         △
External tool integration        ✓         ✗
Simple conditional logic         △         ✓
"""

# YY_RULE_SETUP
# YY_FATAL_ERROR

"""
taken_directive_path: macros with if directives
all_directive_path: macros with if directives including skippped macros

taken_macros_path: all macros 
all_macros_path: all macros including skippped macros
"""

def merge_meta_macros_with_app(target_dir, gen_macro_meta_path, macros_usage_data):
    print("merging macro_defs with usage")
    macro_defs = read_json(gen_macro_meta_path)
    
    #### Usages
    uses = {}
    for item in macro_defs['macros']:
        for used_item in item['uses']:
            ref_key = f"{used_item['name']}:{used_item['definition']}"
            if ref_key not in uses:
                uses[ref_key] = []
            uses[ref_key].append(used_item)

    #### Hold existing macro keys as a set
    existing_keys = set()
    for macro in macros_usage_data['macros']:
        existing_keys.add(f"{macro['name']}:{macro['definition']}")

    #### Add definitions
    for item in macro_defs['macros']:
        ref_key = f"{item['name']}:{item['definition']}"
        if ref_key not in existing_keys:  # ← O(1) lookup
            use_file_path, def_line, def_col = parse_def_loc(item['definition'])
            macros_usage_data['macros'].append({
                "kind": "macro",
                "name": item['name'],
                "definition": item['definition'],
                "file_path": use_file_path,
                "start_line": item['start_line'],
                "end_line": item['end_line'],
                "is_const" : None,
                "is_flag" : None,
                "expanded_value": None,
                "parameters": [],
                "appearances": [],
            })
            existing_keys.add(ref_key)  # Prevent duplicate additions

    #### Merge usages
    for macro in macros_usage_data['macros']:
        macro_key = f"{macro['name']}:{macro['definition']}"
        if macro_key not in uses:
            continue

        uses_list = uses[macro_key]
        for use_item in uses_list:
            app = use_item['usage_location']
            if app not in macro['appearances']:
                macro['appearances'].append(app)
    
    return macros_usage_data


def reform_uses_data(target_dir, macros_usage_data):

    print("merging macro_defs with usage")
    macro_defs = macros_usage_data #read_json(gen_macro_meta_path)
    
    #### Usages
    uses = {}
    for item in macro_defs['macros']:
        for used_item in item['uses']:
            ref_key = f"{used_item['name']}:{used_item['definition']}"
            if ref_key not in uses:
                uses[ref_key] = []
            uses[ref_key].append(used_item)
    
    """
    #### Hold existing macro keys as a set
    existing_keys = set()
    for macro in macros_usage_data['macros']:
        existing_keys.add(f"{macro['name']}:{macro['definition']}")

    #### Add definitions
    for item in macro_defs['macros']:
        ref_key = f"{item['name']}:{item['definition']}"
        if ref_key not in existing_keys:  # ← O(1) lookup
            use_file_path, def_line, def_col = parse_def_loc(item['definition'])
            macros_usage_data['macros'].append({
                "kind": "macro",
                "name": item['name'],
                "definition": item['definition'],
                "file_path": use_file_path,
                "start_line": item['start_line'],
                "end_line": item['end_line'],
                "is_const" : None,
                "is_flag" : None,
                "expanded_value": None,
                "parameters": [],
                "appearances": [],
            })
            existing_keys.add(ref_key)  # Prevent duplicate additions
    """

    #### Merge usages
    for macro in macros_usage_data['macros']:
        macro_key = f"{macro['name']}:{macro['definition']}"
        if macro_key not in uses:
            continue

        uses_list = uses[macro_key]
        for use_item in uses_list:
            app = use_item['usage_location']
            if app not in macro['appearances']:
                macro['appearances'].append(app)
    
    return macros_usage_data


def merge_directive_macros_with_app(target_dir, taken_directive_path, macros_usage_data):
    print("merging directive_macros with usage...")

    directives = read_json(taken_directive_path)
    seen = set()

    # Convert existing macros' appearances to sets for fast lookup
    for macro in macros_usage_data['macros']:
        macro_key = f"{macro['name']}:{macro['definition']}"
        if macro_key not in directives:
            continue

        seen.add(macro_key)

        # Convert list → set (O(n) → O(1) lookup)
        app_set = set(macro['appearances'])
        for use_item in directives[macro_key]['appearances']:
            app = f"{use_item['file_path']}:{use_item['start_line']}:{use_item['start_column']}"
            app_set.add(app)  # Duplicates are automatically excluded
        macro['appearances'] = list(app_set)

    # Bulk-add unmatched entries
    for macro_key, item in directives.items():
        if macro_key in seen:
            continue

        file_path = item['definition']['file_path'] or "undefined"
        start_line = item['definition']['start_line']
        start_column = item['definition']['start_column']

        definition = file_path if start_line is None else f"{file_path}:{start_line}:{start_column}"

        macros_usage_data['macros'].append({
            "kind": "macro",
            "name": item['name'],
            "definition": definition,
            "start_line": start_line,
            "end_line": start_line,
            "file_path": file_path,
            "is_const": None,
            "is_flag": None,
            "expanded_value": None,
            "parameters": [],
            "appearances": [
                f"{a['file_path']}:{a['start_line']}:{a['start_column']}"
                for a in item['appearances']
            ]
        })

    return macros_usage_data


def get_compile_json(target_dir):
    print(target_dir)

    compile_dir = find_compile_commands_json(target_dir)
    if compile_dir is None:
        return None, None

    compile_dir = Path(compile_dir)
    compile_json_path = compile_dir / "compile_commands.json"
    
    print(compile_json_path)
    ########
    compile_commands = read_json(compile_json_path)
    EXCLUDE_EXTENSIONS = {'.S', '.s', '.asm', '.ASM'}
    original_count = len(compile_commands)
    filtered = []
    for e in compile_commands:
        file_path = e.get('file', '')
        is_excluded = False
        for ext in EXCLUDE_EXTENSIONS:
            if file_path.endswith(ext):
                is_excluded = True
                break
        if not is_excluded:
            filtered.append(e)
    compile_commands = filtered

    excluded_count = original_count - len(compile_commands)
    if excluded_count > 0:
        print(f"Filtered out {excluded_count} files with excluded extensions: {EXCLUDE_EXTENSIONS}") 
        write_json(compile_json_path, compile_commands)
    
    ########

    convert_to_absolute_paths(compile_json_path)

    return compile_dir, compile_json_path


def replace_in_value(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)
    elif isinstance(obj, list):
        return [replace_in_value(item, old, new) for item in obj]
    elif isinstance(obj, dict):
        return {k: replace_in_value(v, old, new) for k, v in obj.items()}
    else:
        return obj


def setup_compile_json(given_compile_dir, old_directory, new_directory):  # , given_compile_json_path
    
    given_compile_json_path = f"{given_compile_dir}/compile_commands.json"

    new_compile_dir = given_compile_dir.replace(old_directory, new_directory)
    new_compile_json_path = given_compile_json_path.replace(old_directory, new_directory)

    copy_file(given_compile_json_path, new_compile_dir)

    # Replace old_directory with new_directory in new_compile_json_path
    data = read_json(new_compile_json_path)

    data = replace_in_value(data, old_directory, new_directory)

    with open(new_compile_json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Setup compile json: {new_compile_json_path}")
    print(f"Replaced paths: {old_directory} -> {new_directory}")
    
    return new_compile_json_path  #given_compile_json_path


def clone_compile_json(original_dir, old_dir, new_dir):  #given_compile_dir, old_dir, new_dir):
    given_compile_dir = find_compile_commands_json(original_dir)

    compile_json_path = f"{given_compile_dir}/compile_commands.json"
    new_compile_json_path = compile_json_path.replace(old_dir, new_dir)
    new_compile_dir = os.path.dirname(new_compile_json_path)

    os.makedirs(new_compile_dir, exist_ok=True)

    copy_file(compile_json_path, new_compile_dir)
    # print(compile_json_path)
    compile_commands = read_json(compile_json_path)
    compile_commands = replace_in_value(compile_commands, old_dir, new_dir)
    # setup_compile_json(given_compile_dir, given_compile_json_path, "/home/ubuntu/macrust", "/home/ubuntu/allrust")
    # compile_dir, compile_json_path = given_compile_dir, given_compile_json_path
    write_json(new_compile_json_path, compile_commands)

    print(new_compile_json_path)


def append_compile_json_path(compile_json_path, database_dir): #f"{database_dir}/compile_commands.json")

    database_json_path = f"{database_dir}/compile_commands.json"

    new_entries = read_json(compile_json_path)

    existing_entries = []
    if os.path.exists(database_json_path):
        existing_entries = read_json(database_json_path)

    # Convert existing entries to dict (using file as key)
    merged = {}
    # for entry in existing_entries:
    #     merged[entry["file"]] = entry
    for entry in existing_entries:
        key = (entry["file"], entry.get("output", ""), entry.get("directory", ""))
        merged[key] = entry

    # Overwrite with new entries (last write wins)
    # for entry in new_entries:
    #     merged[entry["file"]] = entry
    for entry in new_entries:
        key = (entry["file"], entry.get("output", ""), entry.get("directory", ""))
        merged[key] = entry

    with open(compile_json_path, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, indent=2, ensure_ascii=False)

    # save_compile_json_path
    copy_file(compile_json_path, database_dir)



def get_global_definition(var_name, global_vars):
    if var_name not in global_vars:
        return None, None, None

    return global_vars[var_name]['file_path'], global_vars[var_name]['start_line'], global_vars[var_name]['start_column']


def detect_global_vars(target_dir, meta_dir, global_path, is_program_path):

    global_vars = {} #[]
    parsed_names = []
    meta_files = get_all_files(meta_dir)
    program_files = set(read_json(is_program_path))

    for meta_path in meta_files:
        meta_data = read_json(meta_path)
        for def_key, item in meta_data.items():
            name = item['name']
            if item['kind'] != 'global_var':
                continue
            #print(item)
            def_file_path, def_start_line, def_start_column = parse_def_loc(item['definition'])

            if is_system_file(def_file_path, program_files):
                continue

            parsed_names.append(name)

            if name not in global_vars:
                global_vars[name] = {}
            global_vars[name] = {
                'file_path' : def_file_path,
                'start_line' : def_start_line,
                'start_column' : def_start_column
            }

    compile_commands_dir = find_compile_commands_json(target_dir)
    cc_path = os.path.join(compile_commands_dir, "compile_commands.json")
    with open(cc_path) as f:
        cc = json.load(f)

    target_dir_abs = os.path.abspath(target_dir)
    if not target_dir_abs.endswith('/'):
        target_dir_abs += '/'

    all_globals = {}

    for entry in cc:
        file_path = entry.get('file', '')

        if not file_path.startswith(target_dir_abs):
            continue

        if file_path.endswith(('.cxx', '.cpp', '.cc')):
            continue

        args = entry.get("arguments", [])
        flags = [arg for arg in args if arg.startswith(("-I", "-D"))]

        cmd = [
            "bindgen", file_path,
            "--clang-macro-fallback",
            "--",
        ] + flags

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"bindgen failed for {file_path}: {result.stderr[:200]}")
            continue

        # Extract global variables from bindgen output
        bindings = result.stdout
        pattern = re.compile(
            #r'pub\s+static\s+mut\s+(\w+)\s*:\s*(.+?)\s*;',
            r'pub\s+static\s+mut\s+(\w+)\s*:\s*(.+?)\s*;\s*$',
            re.MULTILINE
        )
        for m in pattern.finditer(bindings):
            var_name = m.group(1)
            rust_type = m.group(2).strip()

            if var_name in all_globals:
                # If the same variable is found in multiple files, verify type consistency
                if all_globals[var_name]["rust_type"] != rust_type:
                    raise ValueError(f"WARNING: type conflict for '{var_name}': "
                          f"{all_globals[var_name]['rust_type']} vs {rust_type} "
                          f"(from {file_path})")
                continue

            if var_name not in parsed_names:
                continue

            def_file_path, def_start_line, def_start_column = get_global_definition(var_name, global_vars)
            all_globals[var_name] = {
                "var_name": var_name,
                "rust_type": rust_type,
                "definition" : {
                    "file_path" : def_file_path,
                    "start_line" : def_start_line,
                    "start_column" : def_start_column,
                }
            }
            #"source_file": os.path.relpath(file_path, target_dir_abs),

    globals_list = sorted(all_globals.values(), key=lambda x: x["var_name"])
    os.makedirs(os.path.dirname(global_path), exist_ok=True)
    write_json(global_path, globals_list)

    print(f"Detected {len(globals_list)} global variables -> {global_path}")
    return globals_list



def parse_all(round_id, macro_finder, target_dir, meta_dir, div_meta_dir, database_dir, build_path,  # , output_file 
                 taken_directive_path, unordered_taken_directive_path, all_directive_path, dep_json_path, is_program_path,  # unordered_taken_directive_path, 
                 all_macros_path, taken_macros_path, guards_path, guarded_macros_path, independent_path, flag_path, const_path,
                 given_compile_dir, given_compile_json_path, global_path): # , cfg_path
    
    output_file = f"{database_dir}/macro_finder_results.txt"
    
    #unordered_taken_directive_path = f"{database_dir}/output_used.json"
    ordered_all_directive_path = f"{database_dir}/all_output_def.json"

    if round_id in ["1"]: #["1", "2"]:
        option = "init"
    else:
        option = "build" #"both"
    
    if round_id == "call":
        option = ""

    print(f"\n====== Start round {round_id} ======")
    # print(build_path)
    # print(option)

    if round_id == "all":
        given_compie_dir = find_compile_commands_json(target_dir)
        copy_file(given_compile_json_path, f"{database_dir}")


    error_output, std_output = run_script_wo_log(build_path, 10000, True, None, option)
    if error_output is not None:
        raise ValueError(f"Faild to run {build_path} at round {round_id}")
    
    check_permission(target_dir)

    compile_dir, compile_json_path = get_compile_json(target_dir)


    if round_id == "all":
        compile_commands = read_json(compile_json_path)

        print(given_compile_json_path)
        print(compile_json_path)

        if compile_dir is None or len(compile_commands) == 0:
            copy_file(f"{database_dir}/compile_commands.json", given_compie_dir)

            """
            #compile_dir, compile_json_path = given_compile_dir, given_compile_json_path  #denormalize_compile_json(given_compile_json_path, compile_json_path)
            copy_file(given_compile_json_path, compile_dir)
            # print(compile_json_path)
            compile_commands = read_json(compile_json_path)
            compile_commands = replace_in_value(compile_commands, "macrust", "allrust")
            # setup_compile_json(given_compile_dir, given_compile_json_path, "/home/ubuntu/macrust", "/home/ubuntu/allrust")
            # compile_dir, compile_json_path = given_compile_dir, given_compile_json_path
            write_json(compile_json_path, compile_commands)
            """
            compile_commands = read_json(compile_json_path)

            if compile_dir is None or len(compile_commands) == 0:
                raise ValueError("Did not find compile_commands.json") 
         
    else:
        if compile_dir is None:
            raise ValueError("Did not find compile_commands.json")

    # Append with the current compile_commands.json
    append_compile_json_path(compile_json_path, database_dir)

    with ProcessPoolExecutor(max_workers=4) as pool:

        # ---- Step : find_headers || run_finder ----
        futures = {}
        if round_id in ["call", "1", "2", "3", "all"]: # , "all" #, "2", "3"]:
            compile_log_path = f'{database_dir}/compile.log'
            build_dir = find_compile_commands_json(target_dir)
            #analyze_dependencies(target_dir, dep_json_path, build_path, compile_log_path, build_dir, database_dir)
            futures["find_headers"] = pool.submit(find_headers, target_dir, database_dir, dep_json_path, compile_dir, compile_json_path, round_id)
            
            # Wait for find_headers to complete
            if "find_headers" in futures:
                futures["find_headers"].result()
            
            # Execute after dep_json_path has been generated
            generate_is_program(target_dir, dep_json_path, is_program_path)

        futures["run_finder"] = pool.submit(run_finder, macro_finder, target_dir, output_file, compile_dir, compile_json_path, round_id)

        # -- #
        recreate_directory(meta_dir)
        
        # Added here -> needed during rewriting  # This retrieves non-macro elements
        if round_id in ["call", "1", "2", "3"]: #, "3"]:
            macro_on = False
        else:
            macro_on = True

        # ---- Step 3: parsing x2/3 in parallel ----
        # Write tree-parseable code elements to meta_dir
        fut_meta = pool.submit(generate_metadata, macro_on, target_dir, meta_dir, database_dir, compile_dir, compile_json_path, round_id)  # analyzer_path, 
        
        # Preprocessor is a different language so retrieve separately and write to meta_dir: holds pre-expansion info but depends on post-expansion syntax (has the concept of definitions)
        fut_macro_usage = pool.submit(generate_macro_usage_metadata, target_dir, meta_dir, database_dir, independent_path, flag_path, compile_dir, compile_json_path, round_id)
        
        #"""
        gen_macro_meta_path = Path(database_dir) / "meta_macro.json"
        # Insert cases where macro definitions use other code element symbols
        # if round_id in ["call", "all", "4"]:
        #     fut_macro_meta = pool.submit(generate_macro_metadata, target_dir, meta_dir, database_dir, independent_path, compile_dir, compile_json_path) #, independent_path)
        # #"""

        fut_macro_meta = pool.submit(generate_macro_metadata, target_dir, meta_dir, database_dir, independent_path, compile_dir, compile_json_path, round_id) #, independent_path)

        all_symbols, gen_meta_path = fut_meta.result()  # , macro_dep_path
        all_usage_symbols, gen_macro_usage_meta_path, macros_usage_data = fut_macro_usage.result()

        macro_dep_path = update_metadata(all_symbols, meta_dir, database_dir, macro_on)
        update_macro_usage_metadata(all_usage_symbols, meta_dir, independent_path, flag_path, target_dir)

        #if round_id in ["call", "all", "4"]:
        all_macro_symbols, gen_macro_meta_path = fut_macro_meta.result()
        update_macro_metadata(all_macro_symbols, meta_dir)

        
        # Wait for both to complete
        for name, fut in futures.items():
            fut.result()  # Exceptions will be raised here if any

        # ---- Step 3: save_all_directives x2 in parallel ----
        # All directive macros (both inactive and active)
        fut_all = pool.submit(save_all_directives, output_file, all_directive_path, ordered_all_directive_path, database_dir, target_dir, False, False)
        
        # Active directive macros
        #save_taken_directives(output_file, unordered_taken_directive_path, taken_directive_path)
        #save_all_directives(output_file, unordered_taken_directive_path, taken_directive_path, database_dir, True, False)
        fut_taken = pool.submit(save_all_directives, output_file, unordered_taken_directive_path, taken_directive_path, database_dir, target_dir, True, False)


        fut_taken.result()
        fut_all.result()

        """
        # Is this even needed?
        if macro_on is True:
            # Insert cases where macro definitions use other code element symbols (shouldn't this be in generate_macro_metadata() instead?)
            insert_macro_deps(macro_dep_path, target_dir, meta_dir)
        """

    macros_usage_data = reform_uses_data(target_dir, macros_usage_data)

    # Merge usage locations from generate_macro_metadata into macros_usage_data
    macros_usage_data = merge_meta_macros_with_app(target_dir, gen_macro_meta_path, macros_usage_data)

    # Merge ifdef usage locations into macros_usage_data
    macros_usage_data = merge_directive_macros_with_app(target_dir, taken_directive_path, macros_usage_data)

    # Merge appearances into metadata uses
    merge_appearances_with_uses(target_dir, meta_dir, database_dir, macros_usage_data)

    write_json(gen_macro_usage_meta_path, macros_usage_data)

    if round_id == "call":
        return

    # get_all_macros(all_macros_path, gen_meta_path, gen_macro_meta_path) # This can't be obtained by the parser. Well, since it's only used for IF, maybe that's fine

    # Detect include guards (this is also dynamic, so it probably needs to be done before dynamic detection)
    detect_include_guards(all_directive_path, target_dir, meta_dir, guards_path, is_program_path) # taken_macros_path # Before: pre-expansion info  # Using taken could be inaccurate for cases where duplicate includes occur across files

    detect_guarded_macros(all_directive_path, target_dir, guarded_macros_path, is_program_path)  # , meta_dir

    insert_guarded_flag(guarded_macros_path, gen_macro_usage_meta_path)

    # Get is_flag # Write out cfg attribute items here (macros used for conditional compilation) <- exclude include guards
    detect_flag(all_directive_path, flag_path, gen_macro_usage_meta_path)  # , guarded_macros_path
    
    get_taken_macros(taken_directive_path, gen_macro_usage_meta_path, taken_macros_path, database_dir) # gen_meta_path,  # , gen_macro_meta_path

    if round_id == "all":
        detect_global_vars(target_dir, meta_dir, global_path, is_program_path)

    # Making this change will break things in the C world, so this is translation-only code
    """
    llm_answer_path = f"{database_dir}/answer.json"
    check_guards_with_llms(llm_on, meta_dir, target_dir, guarded_macros_path, llm_answer_path, database_dir, 
                           llm_choice, llm_instance, token_path, chat_dir)
    """

    
    outermost_path = f"{database_dir}/outermost.json"
    combine_with_outermost_conditioned_blocks(all_directive_path, outermost_path, guards_path, target_dir, round_id, meta_dir, is_program_path)

    #if div_meta_dir is not None:
    recreate_directory(div_meta_dir)

    # I think metadata should be stored separately for non-grouped and parallel ones.
    #if div_meta_dir is not None:
    parent = os.path.dirname(div_meta_dir)
    copy_directory(meta_dir, parent)

    combine_with_innermost_conditioned_blocks(all_directive_path, target_dir, database_dir, round_id, div_meta_dir, is_program_path)  # , guarded_macros_path

    # result_src = subprocess.run(["find", meta_dir, "-name", "*.json"], capture_output=True, text=True)
    # count_src = len([f for f in result_src.stdout.strip().split("\n") if f])
    # result_dst = subprocess.run(["find", div_meta_dir, "-name", "*.json"], capture_output=True, text=True)
    # count_dst = len([f for f in result_dst.stdout.strip().split("\n") if f])
    # if count_src != count_dst:
    #     src_files = set(os.path.relpath(f, meta_dir) for f in result_src.stdout.strip().split("\n") if f)
    #     dst_files = set(os.path.relpath(f, div_meta_dir) for f in result_dst.stdout.strip().split("\n") if f)
    #     missing = src_files - dst_files
    #     print(f"DEBUG: Files not copied: {len(missing)}")
    #     for f in list(missing)[:10]:
    #         print(f"  - {f}")


    if round_id in ["4"]: #["1", "2", "3", "4"]:
        # detect_flag(unordered_taken_directive_path, guarded_macros_path, flag_path)  #detect_cfg(all_directive_path, guards_path, cfg_path)

        # Incorporate cfg if-statements as component elements to facilitate prompt generation
        #insert_ifdef_statement(cfg_path, target_dir, meta_dir) # flag_path = cfg_path maybe
        insert_ifdef_statement(flag_path, target_dir, meta_dir) # flag_path = cfg_path maybe


    p_f(define_conditioned_blocks, target_dir, True, True, round_id, meta_dir) # # , all_macro_path dep_json_path # blocks.py, # analyze_blocks

    # define blocks
    # define_blocks(round_id, all_directive_path, guarded_macros_path, target_dir, meta_dir, div_meta_dir, database_dir)  # , raw_dir



def replace_macro_func_ref(macro_func_path, target_dir, output_dir):
    print("replace_macro_func_ref()...")



def detect_sys_macros(macro_metadata, sys_macros_path):
    """
    A function that adds the relied_on_sys flag to the JSON
    when the definition of an element in uses is in a system file.
    """
    print("detect_sys_macros()...")
    
    # Path patterns for system headers
    system_include_paths = [
        '/usr/include',
        '/usr/local/include',
        '/usr/lib',
        '/lib',
        '/opt',
        '<built-in>',
        '<command line>'
    ]
    
    def is_program_path(path):
        """Determine whether a path is a system header"""
        if not path or path == "unknown":
            return False
        return any(path.startswith(sys_path) for sys_path in system_include_paths)
    
    # Load macro_metadata
    if isinstance(macro_metadata, str):
        with open(macro_metadata, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = macro_metadata
    
    # Process each macro
    for macro in data.get('macros', []):
        relied_on_sys = False
        
        # Check each element of uses
        if 'uses' in macro:
            for use in macro['uses']:
                definition = use.get('definition', '')
                
                # Check if the definition is in a system file
                if is_program_path(definition):
                    relied_on_sys = True
                    break
        
        # Also check the macro's own definition (optional)
        if is_program_path(macro.get('definition', '')):
            relied_on_sys = True
        
        # Add the flag
        macro['relied_on_sys'] = relied_on_sys
    
    # Save the results
    write_json(sys_macros_path, data)
    
    # Display statistics
    total_macros = len(data.get('macros', []))
    sys_dependent = sum(1 for m in data.get('macros', []) if m.get('relied_on_sys', False))
    
    print(f"Total macros: {total_macros}")
    print(f"System-dependent macros: {sys_dependent}")
    print(f"User-defined macros: {total_macros - sys_dependent}")
    print(f"Results saved to: {sys_macros_path}")
    
    return data


# Write out cfg attribute items here (macros used for conditional compilation) <- Remove include guards
def detect_flag(all_directive_path, flag_path, gen_macro_usage_meta_path): # , guards_path

    print("Detecting flag macros...")
    conds = []

    all_macros = read_json(all_directive_path)
    #guard_macros = read_json(guards_path)
    usage_macros = read_json(gen_macro_usage_meta_path)

    # --- Build a reverse index grouped by name ---
    usage_by_name = {}  # name -> [item, item, ...]
    for item in usage_macros['macros']:
        item['is_flag'] = False
        usage_by_name.setdefault(item['name'], []).append(item)

    #guards = set(item['macro_name'] for item in guard_macros["guards"])

    # --- Scan ifdef / ifndef ---
    def process_macro(item):
        macro_name = item["name"]

        """
        if macro_name not in guards:  # set so O(1)
            conds.append(item)
            # Reverse lookup with O(1)
            for uv in usage_by_name.get(macro_name, []):
                uv['is_flag'] = True
        """

        conds.append(item)
        # Reverse lookup with O(1)
        for uv in usage_by_name.get(macro_name, []):
            uv['is_flag'] = True

    for file_path, data in all_macros.items():
        for directive in ("ifdef", "ifndef"):
            for item in data[directive]:
                if 'name' in item:
                    process_macro(item)
                if 'macros' in item:
                    for each_item in item['macros']:
                        process_macro(each_item)

    write_json(flag_path, conds)  # list() not needed (already a list)
    write_json(gen_macro_usage_meta_path, usage_macros)


def insert_target_annotation(target_dir, target_path, marker):
    print("Inserting target annotation...")
    if not os.path.exists(target_path):
        #return
        raise ValueError(f"We don't find target_path: {target_path}")
    
    with open(target_path, 'r') as f:
        lines = f.readlines()
    
    file_edits = {}
    for line in lines:
        parts = line.strip().split(':')
        if len(parts) < 4:
            continue
        name = parts[0]
        file_path = parts[1]
        line_num = int(parts[2])
        end_line = int(parts[3])
        
        file_path = f'{target_dir}/{file_path}'
        file_path = os.path.abspath(file_path)
        
        if file_path not in file_edits:
            file_edits[file_path] = []
        
        file_edits[file_path].append({
            "start_line": line_num,
            "end_line" : end_line,
            #"column": column,
            "name": name
        })
    
    for file_path, edits in file_edits.items():
        with open(file_path, 'r') as f:
            file_lines = f.readlines()
        
        # Sort in descending order of line number (editing from the bottom prevents line number shifts)
        edits.sort(key=lambda x: x['start_line'], reverse=True)
        
        for edit in edits:
            line_num = edit['start_line']
            #column = edit['column']
            name = edit['name']
            
            idx = line_num - 1
            if idx < 0 or idx >= len(file_lines):
                continue
            
            target_line = file_lines[idx].rstrip('\n')  # Remove newline
            
            # Create marker (added as a comment at end of line)
            annotation = f' /* Genifai: here is one target function!: {file_path}:{line_num}:{name} */'
            new_line = target_line + annotation + '\n'
            
            file_lines[idx] = new_line
        
        with open(file_path, 'w') as f:
            f.writelines(file_lines)
        
        print(f"Updated: {file_path}")



# Determined by Expr::EvaluateAsRValue
def detect_const(const_path, target_dir, meta_dir):
    print("detect_const ...")

    # 1. Generate temporary files with test lines inserted right after each TU's use site
    #    Return value: { tu_path: { test_line_number: (macro_name, def_id) } }
    test_line_map = insert_volatile(target_dir, meta_dir)

    # 2. Load compile_commands.json
    compile_commands_dir = find_compile_commands_json(target_dir)
    cc_path = os.path.join(compile_commands_dir, "compile_commands.json")
    with open(cc_path) as f:
        cc = json.load(f)

    target_dir_abs = os.path.abspath(target_dir)
    if not target_dir_abs.endswith('/'):
        target_dir_abs += '/'

    # 3. Run clang -fsyntax-only for each TU
    error_lines = {}  # { tu_path: set of error line numbers }

    for entry in cc:
        file_path = entry.get('file', '')
        if not file_path.startswith(target_dir_abs):
            continue
        if file_path not in test_line_map:
            continue

        args = entry.get("arguments", [])
        flags = [arg for arg in args if arg.startswith(("-I", "-D"))]

        cmd = [
            "clang", "-fsyntax-only", "-ferror-limit=0",
            file_path
        ] + flags

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Extract error line numbers
        error_lines[file_path] = set()
        for line in result.stderr.splitlines():
            m = re.match(rf'^{re.escape(file_path)}:(\d+):\d+: error:', line)
            if m:
                error_lines[file_path].add(int(m.group(1)))

    # 4. Aggregate the results of test lines
    consts = {}  # { (macro_name, def_id): is_const }

    for tu_path, line_map in test_line_map.items():
        tu_errors = error_lines.get(tu_path, set())
        for test_line, (macro_name, def_id) in line_map.items():
            key = f"{macro_name}:{def_id}"
            is_const = test_line not in tu_errors
            if key not in consts:
                consts[key] = is_const
            else:
                # If judgments differ across multiple use sites, false if even one is false
                consts[key] = consts[key] and is_const

    write_json(const_path, consts)

    # 6. Remove test lines and restore to original
    remove_volatile(target_dir, meta_dir)

    print(f"detect_const done: {len(consts)} definitions checked")


# cargo install bindgen-cli
def detect_const_binden(const_path, target_dir):
    print("get_is_const ...")

    compile_commands_dir = find_compile_commands_json(target_dir)
    cc_path = os.path.join(compile_commands_dir, "compile_commands.json")
    with open(cc_path) as f:
        cc = json.load(f)

    target_dir_abs = os.path.abspath(target_dir)
    if not target_dir_abs.endswith('/'):
        target_dir_abs += '/'

    consts = {}

    for entry in cc:
        file_path = entry.get('file', '')

        # Only files within target_dir
        if not file_path.startswith(target_dir_abs):
            continue

        # Skip C++
        if file_path.endswith(('.cxx', '.cpp', '.cc')):
            continue

        # Get only the flags for this entry
        args = entry.get("arguments", [])
        flags = [arg for arg in args if arg.startswith(("-I", "-D"))]

        cmd = [
            "bindgen", file_path,
            "--clang-macro-fallback",
            "--",
        ] + flags

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"bindgen failed for {file_path}: {result.stderr[:200]}")
            continue

        pattern = r'pub const (\w+):\s*([^=]+)\s*=\s*([^;]+);'
        for match in re.finditer(pattern, result.stdout):
            name = match.group(1)
            rust_type = match.group(2).strip()
            value = match.group(3).strip()

            is_integer = rust_type in [
                'i8', 'i16', 'i32', 'i64', 'i128',
                'u8', 'u16', 'u32', 'u64', 'u128',
                'isize', 'usize', 'c_int', 'c_uint',
                'c_long', 'c_ulong', 'c_longlong', 'c_ulonglong'
            ]
            is_float = rust_type in ['f32', 'f64']
            is_string = 'c_char' in rust_type or rust_type.startswith('&')

            if name not in consts:
                consts[name] = []

            consts[name].append({
                'file_path': file_path,
                'flags': flags,
                'is_const': True,
                'is_integer': is_integer,
                'is_float': is_float,
                'is_string': is_string,
                'rust_type': rust_type,
                'value': value
            })

    write_json(const_path, consts)


def insert_const(const_path, target_dir, database_dir, meta_dir):

    consts = read_json(const_path)

    # Merge into meta_dir (same as existing code)
    meta_files = get_all_files(meta_dir)
    for meta_path in meta_files:
        meta_data = read_json(meta_path)
        for nema_key, item in meta_data.items():
            if 'name' in item and item['name'] in consts:
                item['is_const'] = True
            if 'components' in item:
                for each_key, each_item in item['components'].items():
                    if 'name' in each_item and each_item['name'] in consts:
                        each_item['is_const'] = True
        write_json(meta_path, meta_data)


def apply_replacements_to_file(file_path, replacements):
    """Replace macro usage locations in the file with their expanded values"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # If multiple replacements are on the same line, process in descending column order (replace from the end to prevent offset shifts)
    replacements = sorted(replacements, key=lambda x: (x['line'], x['column']), reverse=True)

    for rep in replacements:
        line_idx = rep['line'] - 1
        if line_idx < 0 or line_idx >= len(lines):
            print(f"Warning: Invalid line {rep['line']} in {file_path}")
            continue

        line = lines[line_idx]
        col = rep['column'] - 1  # 0-indexed
        name = rep['name']

        # Verify the macro name exists at the column position and replace
        if line[col:col + len(name)] == name:
            lines[line_idx] = line[:col] + rep['expanded_text'] + line[col + len(name):]
        else:
            # Fallback if the column position doesn't match
            lines[line_idx] = line.replace(name, rep['expanded_text'], 1)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Applied {len(replacements)} replacements to {file_path}")


def merge_ranges(ranges):
    """
    Merge overlapping or contiguous ranges.
    Result is in descending order (for deleting from the end).
    """
    if not ranges:
        return []
    
    sorted_ranges = sorted(ranges, key=lambda x: x['start'])
    merged = [sorted_ranges[0].copy()]
    
    for current in sorted_ranges[1:]:
        last = merged[-1]
        # Merge if overlapping or contiguous
        if current['start'] <= last['end'] + 1:
            last['end'] = max(last['end'], current['end'])
        else:
            merged.append(current.copy())
    
    # Return in descending order (for deleting from the end)
    merged.sort(key=lambda x: x['start'], reverse=True)
    return merged



def setup_macro_without_transforming(llm_on, macro_finder, target_dir, database_dir, meta_dir, div_meta_dir, build_path, cfg_path, target_path, marker,  
                            list_path, dep_json_path, custom_headers_dir, custom_json_path, custom_header_path, # c_run_path, call_path, # picked_path, macro_list_path, macro_path, all_macro_path, # classified_path, build_rs_path, # defined_path, undefined_path, cmd_line_path, 
                            llm_choice, llm_instance, token_path, chat_dir, all_macros_path, taken_macros_path, 
                            all_directive_path, taken_directive_path, is_program_path, global_path,
                            guards_path, guarded_macros_path, independent_path, flag_path, const_path, conflict_path  # , sys_macros_path
                            ):
    
    print("Setting up conditional macros ...")
    all_output_file = f"{database_dir}/macro_finder_results_all.txt"

    #output_json_path = f"{database_dir}/output.json"
    # all_directive_path = f"{database_dir}/directive_used.json"
    # taken_directive_path = f"{database_dir}/all_directive_def.json"
    unordered_taken_directive_path = f"{database_dir}/unordered_taken_directive.json"

    dynamic_path = f"{database_dir}/dynamic_macros.json"
    #all_dynamic_path = f"{database_dir}/dynamic_macros_all.json"
    program_dynamic_path = f"{database_dir}/dynamic_macros_program.json"
    dynamic_flag_path = f"{database_dir}/dynamic_flag.json"  # f"{database_dir}/dynamic_cond.json"
    dynamic_const_path = f"{database_dir}/dynamic_const.json"

    if_path = f"{database_dir}/if_macros.json"
    macro_func_path = f"{database_dir}/macro_func.json"

    # conditioned_path = f"{database_dir}/conditioned_macros.json"
    print(target_dir)
    output_dir = target_dir #output_dir = f"test_out" #output_dir = target_dir # Production
    #recreate_directory(output_dir)

    # Insert a macro to record the initial position
    insert_target_annotation(target_dir, target_path, marker)

    # Extract comments from all files once <-Is this functioning?
    target_tmp = tmp_backup_directory(target_dir)
    p_f(replace_comments_with_spaces_file, target_dir, True, True)

    #---------------------------------------------
    """
    # This is quite advanced, so let's skip it
    # get possible states
    get_possible_states("1", macro_finder, target_dir, meta_dir, div_meta_dir, database_dir, build_path, 
                 taken_directive_path, all_directive_path, dep_json_path, 
                 all_macros_path, taken_macros_path, guards_path, independent_path)
    """

    # remove_if_defined(target_dir, database_dir, compile_dir, meta_dir, macro_finder, dep_json_path, compile_json_path, build_path, round_id)

    #---------------------------------------------
    # 1st round: parsing # Line numbers change if not split into multiple rounds
    parse_all("1", macro_finder, target_dir, meta_dir, div_meta_dir, database_dir, build_path, 
                 taken_directive_path, unordered_taken_directive_path, all_directive_path, dep_json_path, is_program_path, 
                 all_macros_path, taken_macros_path, guards_path, guarded_macros_path, independent_path, flag_path, const_path,
                 None, None, global_path)  # , cfg_path



def insert_c_code(file_path, raw_dir, meta_dir):
    meta_data, meta_path = obtain_metadata(file_path, meta_dir, False, None, "def")

    if meta_data is None:
        return
    
    for item in meta_data:
        item['c_code'] = read_specific_lines(item['file_path'], item['start_line'], item['end_line'])
    
    write_json(meta_path, meta_data)



def c_create_usedata(file_path, raw_dir, meta_dir, call_path, list_path):
    print(f"Creating c_use data: {file_path}")

    order = get_files_list(list_path)
    if file_path not in order:
        return

    # 1. Load call_data and undefined
    call_data = read_json(call_path)
    undefined_data = read_json("previous_cfgs_undefined.json")  # or undefined_path
    
    # 2. Obtain use_meta_data (existing "use" metadata)
    use_meta_data, use_meta_path = obtain_metadata(file_path, meta_dir, False, None, "use")
    if use_meta_data is None:
        use_meta_data = []
    
    #
    # --- (A) First, add call_data to use_meta_data ---
    #
    for call in call_data:
        if call['call_file_path'] != file_path:
            continue
        text = read_specific_lines(file_path, call['call_start_line'], call['call_start_line'])
        use_meta_data.append({
            "name": call['name'],
            "kind": call['kind'],
            "file_path": file_path,
            "start_line": call['call_start_line'],
            "end_line": call['call_start_line'],
            "source_path": call['def_file_path'],
            "element_id": None,
            "context": text,
            "o_file_path": file_path,
            "o_start_line": call['call_start_line'],
            "o_end_line": call['call_start_line'],
        })
    
    #
    # --- (B) Add undefined_data to use_meta_data ---
    #
    for macro_file_path, macros in undefined_data.items():
        for macro_name, macro_info in macros.items():
            for item in macro_info['usages']:
                if item['file_path'] == file_path:
                    call_start_line = item['start_line']
                    text = read_specific_lines(file_path, call_start_line, call_start_line)
                    use_meta_data.append({
                        "name": item["name"],
                        "kind": "macro_var",
                        "file_path": file_path,
                        "start_line": call_start_line,
                        "end_line": call_start_line,
                        "source_path": "temporary.c",
                        "element_id": None,
                        "context": text,
                        "o_file_path": file_path,
                        "o_start_line": call_start_line,
                        "o_end_line": call_start_line,
                    })

    #
    # --- (C) If writing only once, skip writing here for now ---
    #     Or you can write temporarily here; writing all at once at the end is also OK
    #
    # write_json(use_meta_path, use_meta_data)
    
    #
    # --- (D) Set element_id all at once ---
    #
    # 1) Get the list of includable files from the dependencies associated with file_path
    sum_include_files = get_both_dep(file_path, dep_json_path)
    
    # 2) For each file, load the "def" metadata and build a name->element_id map
    #    How to handle duplicates with the same name needs consideration,
    #    but for now we adopt the "first one found" approach
    name_to_element_id = {}
    for inc_file in sum_include_files:
        meta_def_data, meta_def_path = obtain_metadata(inc_file, meta_dir, False, None, "def")
        if meta_def_data is None:
            continue
        
        for def_item in meta_def_data:
            nm = def_item['name']
            # Only set if not already registered
            if nm not in name_to_element_id:
                name_to_element_id[nm] = def_item['element_id']
    
    # 3) For each element in use_meta_data, look up from name_to_element_id
    global current_element_id
    for use_item in use_meta_data:
        nm = use_item['name']
        if nm in name_to_element_id:
            use_item['element_id'] = name_to_element_id[nm]
        else:
            # If not found, issue a new one
            use_item['element_id'] = current_element_id
            current_element_id += 1
    
    # 4) Write all at once, only once
    write_json(use_meta_path, use_meta_data)

    merge_macro_usage(file_path, meta_dir, macro_path)
    #merge_macro_usage(file_path, meta_dir, all_macro_path)
    #c_add_macro_usage(source_path, meta_dir, all_macro_path)



def get_ref_files(c_path, dep_json_path):  # , c_lib_path, c_build_path, c_cargo_path
    include_files = []
    dep_json = read_json(dep_json_path)

    found = False
    for dep_item in dep_json:
        source_path = dep_item['source']
        if c_path == source_path:
            include_files = dep_item['indirect_include']
            found = True
        else:
            if 'div_parts' in dep_item:
                div_parts = dep_item['div_parts']
                for part in div_parts:
                    if c_path == part['source']:
                        found = True
                        include_files = part['include']
        if found:
            break

    # include_files.append(c_lib_path)
    # include_files.append(c_build_path)
    # include_files.append(c_cargo_path)

    return include_files


def c_reform_usedata(file_path, raw_dir, meta_dir, dep_json_path): # , c_lib_path, c_build_path, c_cargo_path

    # For use data, search for types missing source information (needed for function signatures)
    use_meta_data, use_meta_path = obtain_metadata(file_path, meta_dir, False, None, "use")
    if use_meta_data is None:
        return

    include_files = get_ref_files(file_path, dep_json_path)  # , c_lib_path, c_build_path, c_cargo_path
    include_files.append(file_path)

    filtered_data = []
    for item in use_meta_data:
        if item['source_path'] is None:
            source_path = None
            element_id = None
            for include_path in include_files:
                meta_data, meta_path = obtain_metadata(include_path, meta_dir, False, None, "def")
                if meta_data is None:
                    continue
                for entry in meta_data:
                    if item != entry and item['name'] == entry['name']:
                        source_path = include_path
                        element_id = entry['element_id']
                        break
            if source_path is not None:
                item['source_path'] = source_path
                item['element_id'] = element_id

                filtered_data.append(item)
        else:
            filtered_data.append(item)
    
    write_json(use_meta_path, filtered_data)
    #write_json(use_meta_path, use_meta_data)



def process_find_headers(entry, analyzer_path, compile_dir):
    """Function to process a single file (for parallel execution)"""
    
    source_file = entry.get('file')
    if not source_file:
        return {
            'source_file': None,
            'includes': None,
            'error': None,
            'skip': True
        }
    
    # Skip assembly files
    if source_file.endswith(('.S', '.s', '.asm', '.ASM')):
        return {
            'source_file': source_file,
            'includes': None,
            'error': None,
            'skip': True
        }
    
    directory = entry.get('directory', str(compile_dir))
    
    # Get compile arguments
    arguments = entry.get('arguments', [])
    if not arguments:
        command = entry.get('command', '')
        arguments = command.split()
    
    source_file_abs = source_file
    compile_args = []
    skip_next = False
    
    if len(arguments) > 1:
        for arg in arguments[1:]:
            if skip_next:
                skip_next = False
                continue
            
            if arg == '-o':
                skip_next = True
                continue
            
            if not os.path.isabs(arg):
                arg_abs = os.path.normpath(os.path.join(directory, arg))
            else:
                arg_abs = os.path.normpath(arg)
            
            if arg_abs != source_file_abs and not arg.endswith('.o'):
                compile_args.append(arg)
    
    cmd = [analyzer_path, source_file, "--"] + compile_args
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=directory
    )
    
    if result.returncode == 0:
        if result.stdout and result.stdout.strip():
            try:
                file_metadata = json.loads(result.stdout)
                file_includes = file_metadata.get('includes', [])
                return {
                    'source_file': source_file,
                    'includes': file_includes,
                    'error': None,
                    'skip': False
                }
            except json.JSONDecodeError as e:
                return {
                    'source_file': source_file,
                    'includes': None,
                    'error': {
                        'type': 'json_parse_error',
                        'message': str(e)
                    },
                    'skip': False
                }
        else:
            return {
                'source_file': source_file,
                'includes': [],
                'error': None,
                'skip': False
            }
    else:
        return {
            'source_file': source_file,
            'includes': None,
            'error': {
                'type': 'process_error',
                'returncode': result.returncode,
                'stderr': result.stderr,
                'stdout': result.stdout,
                'cmd': ' '.join(cmd),
                'cwd': directory
            },
            'skip': False
        }



def find_headers(target_dir, database_dir, dep_json_path, compile_dir, compile_json, round_id):
    analyzer_path = "/home/ubuntu/c_parser/include_finder/build/analyzer" 

    # Path normalization
    """
    compile_dir = find_compile_commands_json(target_dir)
    compile_dir = Path(compile_dir)
    compile_json = compile_dir / "compile_commands.json"
    print(compile_json)
    """

    # Check if compile_commands.json exists
    if not compile_json.exists():
        raise FileNotFoundError(
            f"compile_commands.json not found at: {compile_json}"
        )
    
    # Check if the analyzer tool exists
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(
            f"Analyzer tool not found at: {analyzer_path}\n"
            "Please build the analyzer first."
        )
    
    # Load compile_commands.json
    with open(compile_json, 'r') as f:
        compile_commands = json.load(f)
    
    # print(f"Processing {len(compile_commands)} files...")
    
    all_includes = []
    failed_files = []
    
    # Parallel processing settings
    max_workers = min(cpu_count(), 8)
    
    print(f"Processing {len(compile_commands)} files with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_find_headers, entry, analyzer_path, str(compile_dir)): i
            for i, entry in enumerate(compile_commands)
        }
        
        results = {}
        
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()

    # Output processing results in the original order
    for i in range(len(compile_commands)):
        if i not in results:
            continue
            
        res = results[i]
        source_file = res['source_file']
        
        if res['skip'] or source_file is None:
            continue
        
        print(f"[{i+1}/{len(compile_commands)}] Processing {source_file}...")
        
        if res['error'] is None and res['includes'] is not None:
            all_includes.extend(res['includes'])
        
        elif res['error'] is not None:
            error = res['error']
            
            if error['type'] == 'json_parse_error':
                print(f"  ⚠️  JSON parse error: {error['message']}")
                failed_files.append(source_file)
            
            elif error['type'] == 'process_error':
                failed_files.append(source_file)
                print(f"  ❌  Failed (exit code: {error['returncode']})")
                print(f"  STDERR: {error['stderr']}")
                print(f"  STDOUT: {error['stdout']}")
                print(f"  Command: {error['cmd']}")
                print(f"  CWD: {error['cwd']}")
                raise ValueError(f"  ❌  Error: {error['stderr']}")

    # Summary of processing results
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed in finde_headers: {len(compile_commands) - len(failed_files)}/{len(compile_commands)} @round_id {round_id}")
    print(f"📊 Total includes collected: {len(all_includes)}")
    
    if failed_files:
        print(f"❌ Failed files: {len(failed_files)}")
        for f in failed_files[:10]:
            print(f"  - {f}")
    print(f"{'='*60}\n")
    
    # Create the overall metadata
    metadata = {'includes': all_includes}
    
    # Create the database directory
    os.makedirs(database_dir, exist_ok=True)
    database_path = Path(database_dir) / "header.json"
    write_json(str(database_path), metadata)
    print(f"Saved header metadata to: {database_path}")
    

    # Below this, save header.json to dep_json_path in a different format
    # [
    # {
    #     "including_file": "/home/ubuntu/macrust/trans_c_0000/libjpeg-turbo-2.1.0/jfdctflt.c",
    #     "included_file": [
    #         "/home/ubuntu/macrust/trans_c_0000/libjpeg-turbo-2.1.0/jinclude.h:line:column",
    #         "/home/ubuntu/macrust/trans_c_0000/libjpeg-turbo-2.1.0/jpeglib.h:column",
    #         "/home/ubuntu/macrust/trans_c_0000/libjpeg-turbo-2.1.0/jdct.h:column",
    # },

    # Group by including_file
    deps_by_file = {}
    for inc in all_includes:
        including = inc.get('including_file', '')
        included = inc.get('included_file', '')
        line = inc.get('line', 0)
        column = inc.get('column', 0)
        
        if not including:
            continue
        
        if including not in deps_by_file:
            deps_by_file[including] = []
        
        # Add in "file_path:line:column" format
        deps_by_file[including].append(f"{included}:{line}:{column}")

    # Convert to list format
    dep_list = []
    for including_file, included_files in deps_by_file.items():
        dep_list.append({
            "source": including_file,
            "include": list(set(included_files))
        })

    # Build included_by from dep_list
    included_by = {}
    for entry in dep_list:
        source = entry["source"]
        for inc in entry["include"]:
            # Extract file_path from "file_path:line:column"
            parts = inc.rsplit(":", 2)
            if len(parts) >= 3:
                included_file = parts[0]
                line = parts[1]
                column = parts[2]
            else:
                included_file = inc
                line = "0"
                column = "0"
            
            if included_file not in included_by:
                included_by[included_file] = []
            
            # Add the information of the including side
            included_by[included_file].append(f"{source}:{line}:{column}")

    # Add included_by to each entry
    for entry in dep_list:
        source = entry["source"]
        if source in included_by:
            entry["included_by"] = list(set(included_by[source]))
        else:
            entry["included_by"] = []
    
    # Also add files that only appear in included_by (header-only files)
    all_sources = set(e["source"] for e in dep_list)
    for included_file, includers in included_by.items():
        if included_file not in all_sources:
            dep_list.append({
                "source": included_file,
                "include": [],
                "included_by": list(set(includers))
            })
    
    # Make sure to include all source files in compile_commands.json in dep_list.
    all_sources = set(e["source"] for e in dep_list)
    for entry in compile_commands:
        src = entry.get("file", "")
        if src and src not in all_sources:
            dep_list.append({
                "source": src,
                "include": [],
                "included_by": []
            })
            all_sources.add(src)

    # Save the dependency JSON
    #dep_json_path = Path(database_dir) / "header_deps.json"
    write_json(str(dep_json_path), dep_list)
    print(f"Saved header dependencies to: {dep_json_path}")


    return metadata, database_path



def process_generate_metadata(entry, analyzer_path, compile_dir):
    """Function to process a single file (faithfully reproducing the original processing)"""
    
    source_file = entry.get('file')
    if not source_file:
        return {
            'source_file': None,
            'symbols': None,
            'error': None,
            'skip': True
        }
    
    # Skip assembly files
    if source_file.endswith(('.S', '.s', '.asm', '.ASM')):
        return {
            'source_file': source_file,
            'symbols': None,
            'error': None,
            'skip': True
        }
    
    directory = entry.get('directory', str(compile_dir))
    
    # Get compile arguments
    arguments = entry.get('arguments', [])
    if not arguments:
        command = entry.get('command', '')
        arguments = command.split()
    
    source_file_abs = source_file
    compile_args = []
    skip_next = False
    
    if len(arguments) > 1:
        for arg in arguments[1:]:
            if skip_next:
                skip_next = False
                continue
            
            if arg == '-o':
                skip_next = True
                continue
            
            if not os.path.isabs(arg):
                arg_abs = os.path.normpath(os.path.join(directory, arg))
            else:
                arg_abs = os.path.normpath(arg)
            
            if arg_abs != source_file_abs and not arg.endswith('.o'):
                compile_args.append(arg)
    
    cmd = [analyzer_path, source_file, "--"] + compile_args
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=directory
    )
    
    if result.returncode == 0:
        if result.stdout and result.stdout.strip():
            try:
                file_metadata = json.loads(result.stdout)
                file_symbols = file_metadata.get('symbols', [])
                return {
                    'source_file': source_file,
                    'symbols': file_symbols,
                    'error': None,
                    'skip': False
                }
            except json.JSONDecodeError as e:
                # Return detailed error information
                lines = result.stdout.split('\n')
                error_line = e.lineno - 1
                
                start = max(0, error_line - 3)
                end = min(len(lines), error_line + 4)
                context_lines = []
                for i in range(start, end):
                    marker = ">>>" if i == error_line else "   "
                    line_content = lines[i][:150] if len(lines[i]) > 150 else lines[i]
                    context_lines.append(f"  {marker} L{i+1}: {line_content}")
                
                return {
                    'source_file': source_file,
                    'symbols': None,
                    'error': {
                        'type': 'json_parse_error',
                        'message': str(e),
                        'lineno': e.lineno,
                        'colno': e.colno,
                        'context': context_lines,
                        'stdout': result.stdout
                    },
                    'skip': False
                }
        else:
            return {
                'source_file': source_file,
                'symbols': [],
                'error': None,
                'skip': False
            }
    else:
        return {
            'source_file': source_file,
            'symbols': None,
            'error': {
                'type': 'process_error',
                'returncode': result.returncode,
                'stderr': result.stderr,
                'stdout': result.stdout,
                'cmd': ' '.join(cmd),
                'cwd': directory
            },
            'skip': False
        }



def generate_metadata(macro_on, target_dir, meta_dir, database_dir, compile_dir, compile_json, round_id):
    """
    Function that specifies the directory containing compile_commands.json,
    batch-runs the tool created with analyzer.cpp, and retrieves metadata (JSON).
    """
    print("Starting generate_metadata...")

    analyzer_path = "/home/ubuntu/c_parser/c_analyzer/analyzer" 
    analyzer_path = "/home/ubuntu/c_parser/c_analyzer/build/analyzer" 
    analyzer_path = "/home/ubuntu/c_parser/test/build/analyzer"
    analyzer_path = "/home/ubuntu/c_parser/usage_analyzer/build/analyzer"


    # Path normalization
    """
    compile_dir = find_compile_commands_json(target_dir)
    compile_dir = Path(compile_dir)
    compile_json = compile_dir / "compile_commands.json"
    print(compile_json)
    convert_to_absolute_paths(compile_json)  # This is necessary!
    """
    compile_json = Path(compile_json)

    # Check if compile_commands.json exists
    if not compile_json.exists():
        raise FileNotFoundError(
            f"compile_commands.json not found at: {compile_json}"
        )
    
    # Check if the analyzer tool exists
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(
            f"Analyzer tool not found at: {analyzer_path}\n"
            "Please build the analyzer first."
        )

    
    # Load compile_commands.json (for summary)
    compile_commands = read_json(compile_json)

    """
    EXCLUDE_EXTENSIONS = {'.S', '.s', '.asm', '.ASM'}
    original_count = len(compile_commands)
    compile_commands = [
        e for e in compile_commands
        if not any(e.get('file', '').endswith(ext) for ext in EXCLUDE_EXTENSIONS)
    ]

    excluded_count = original_count - len(compile_commands)
    if excluded_count > 0:
        print(f"Filtered out {excluded_count} files with excluded extensions: {EXCLUDE_EXTENSIONS}")
        with open(compile_json, 'w') as f:
            json.dump(compile_commands, f, indent=2)
    """
    print(f"Processing {len(compile_commands)} files (batch mode)...")

    # === Plan B: Batch execution ===
    cmd = [analyzer_path, "-p", str(compile_dir)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True, #check=False,
        cwd=str(compile_dir)
    )

    """
    if result.returncode != 0:
        print(f"❌ Analyzer failed (exit code: {result.returncode})")
        print(f"STDERR: {result.stderr}")
        print(f"Command: {' '.join(cmd)}")
        raise ValueError(f"❌ Analyzer error: {result.stderr}")
    """

    if result.returncode != 0:
        print(f"\n=== Error processing {compile_dir} ===")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"Stderr:\n{result.stderr}")
        print(f"Stdout:\n{result.stdout[:500]}")
        # Stop on error
        raise subprocess.CalledProcessError(
            result.returncode, 
            cmd, 
            result.stdout, 
            result.stderr
        )


    # Parse JSON output
    all_symbols = []
    failed_files = []

    if result.stdout.strip():
        try:
            file_metadata = json.loads(result.stdout)
            all_symbols = file_metadata.get('symbols', [])
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Error at line {e.lineno}, column {e.colno}")

            lines = result.stdout.split('\n')
            error_line = e.lineno - 1
            start = max(0, error_line - 3)
            end = min(len(lines), error_line + 4)
            print("--- Context ---")
            for i in range(start, end):
                marker = ">>>" if i == error_line else "   "
                line_content = lines[i][:150] if len(lines[i]) > 150 else lines[i]
                print(f"  {marker} L{i+1}: {line_content}")
            print("---------------")

            debug_path = "/tmp/debug_batch_output.json"
            with open(debug_path, 'w') as f:
                f.write(result.stdout)
            print(f"Debug saved: {debug_path}")

            raise ValueError(f"❌ JSON parse error in batch output")

    # Summary of processing results
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed in generate_metadata (batch mode) @round {round_id}")
    print(f"📊 Total symbols collected: {len(all_symbols)}")
    print(f"{'='*60}\n")

    # Create the overall metadata
    metadata = {'symbols': all_symbols}

    # Create the database directory
    os.makedirs(database_dir, exist_ok=True)
    database_path = Path(database_dir) / "meta.json"
    write_json(str(database_path), metadata)
    print(f"Saved global metadata to: {database_path}")

    """
    # Classify symbols by file
    file_symbols = defaultdict(list)
    macro_symbols = defaultdict(list)

    for item in all_symbols:
        if macro_on is False:
            if 'macro' in item.get('kind'):
                continue

        definition = item.get('definition', '')
        start_line = item.get('start_line', None)
        end_line = item.get('end_line', None)

        # Extract file path from definition
        if ':' in definition:
            parts = definition.split(':')
            if len(parts) >= 3:
                file_path = parts[0]

                if not os.path.isabs(file_path):
                    file_path = os.path.join(str(compile_dir), file_path)

                file_path = os.path.normpath(file_path)

                try:
                    line_num = int(parts[1])
                    col_num = int(parts[2])
                except (ValueError, IndexError):
                    line_num = 0
                    col_num = 0

                use_data = item.get('uses', [])
                for use_item in use_data:
                    if 'definition' in use_item:
                        def_file_path, def_line, def_col = parse_def_loc(use_item['definition'])
                        use_item['file_path'] = def_file_path
                        use_item['start_line'] = def_line

                meta_item = {
                    'kind': item.get('kind', 'unknown'),
                    'name': item.get('name', ''),
                    'definition': definition,
                    'start_line': start_line,
                    'start_column': col_num,
                    'end_line': end_line,
                    'block_start': int(start_line),
                    'block_end': int(end_line),
                    'rust_code': {
                        'file_path': None,
                        'start_line': None,
                        'content': None,
                    },
                    'uses': use_data
                }
                if item.get('kind') == 'function':
                    meta_item['signature'] = item.get('signature', '')

                if macro_on is True:
                    if 'macro' in item.get('kind'):
                        macro_symbols[file_path].append(meta_item)
                    else:
                        file_symbols[file_path].append(meta_item)
                else:
                    file_symbols[file_path].append(meta_item)

    # Save metadata per file
    print(f"\nSaving per-file metadata...")
    for file_path, symbols in file_symbols.items():
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping non-existent file: {file_path}")
            continue

        meta_path = obtain_metadata(file_path, meta_dir, False, True, "def")
        meta_path = Path(meta_path)

        if not os.path.exists(meta_path):
            existing_data = {}
        else:
            existing_data = read_json(meta_path)

        for symbol in symbols:
            name = symbol['name']
            s_line = symbol['start_line']
            item_key = f"{name}:{file_path}:{s_line}"

            if item_key not in existing_data:
                existing_data[item_key] = symbol

        write_json(meta_path, existing_data)

    macro_dep_path = f"{database_dir}/macro_deps.json"
    if macro_on is True:
        write_json(macro_dep_path, macro_symbols)

    print(f"\n{'='*60}")
    print(f"📁 Total files with metadata: {len(file_symbols)}")
    print(f"📊 Total symbols: {sum(len(syms) for syms in file_symbols.values())}")
    print(f"{'='*60}\n")
    """

    return all_symbols, database_path  #metadata, database_path, macro_dep_path


def update_metadata(all_symbols, meta_dir, database_dir, macro_on):

    # Classify symbols by file
    file_symbols = defaultdict(list)
    macro_symbols = defaultdict(list)

    for item in all_symbols:
        if macro_on is False:
            if 'macro' in item.get('kind'):
                continue

        definition = item.get('definition', '')
        start_line = item.get('start_line', None)
        end_line = item.get('end_line', None)

        # Extract file path from definition
        if ':' in definition:
            parts = definition.split(':')
            if len(parts) >= 3:
                file_path = parts[0]

                if not os.path.isabs(file_path):
                    compile_dir = os.path.abspath(compile_dir)
                    file_path = os.path.join(str(compile_dir), file_path)

                file_path = os.path.normpath(file_path)

                try:
                    line_num = int(parts[1])
                    col_num = int(parts[2])
                except (ValueError, IndexError):
                    line_num = 0
                    col_num = 0

                use_data = item.get('uses', [])
                for use_item in use_data:
                    if 'definition' in use_item:
                        def_file_path, def_line, def_col = parse_def_loc(use_item['definition'])
                        use_item['file_path'] = def_file_path
                        use_item['start_line'] = def_line

                meta_item = {
                    'kind': item.get('kind', 'unknown'),
                    'name': item.get('name', ''),
                    'definition': definition,
                    'start_line': start_line,
                    'start_column': col_num,
                    'end_line': end_line,
                    'block_start': int(start_line),
                    'block_end': int(end_line),
                    'rust_code': {
                        'file_path': None,
                        'start_line': None,
                        'content': None,
                    },
                    'uses': use_data
                }
                if item.get('kind') == 'function':
                    meta_item['signature'] = item.get('signature', '')

                if macro_on is True:
                    if 'macro' in item.get('kind'):
                        macro_symbols[file_path].append(meta_item)
                    else:
                        file_symbols[file_path].append(meta_item)
                else:
                    file_symbols[file_path].append(meta_item)

    # Save metadata per file
    print(f"\nSaving per-file metadata...")
    for file_path, symbols in file_symbols.items():
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping non-existent file: {file_path}")
            continue

        meta_path = obtain_metadata(file_path, meta_dir, False, True, "def")
        meta_path = Path(meta_path)

        if not os.path.exists(meta_path):
            existing_data = {}
        else:
            existing_data = read_json(meta_path)

        for symbol in symbols:
            name = symbol['name']
            s_line = symbol['start_line']
            item_key = f"{name}:{file_path}:{s_line}"

            if item_key not in existing_data:
                existing_data[item_key] = symbol

        write_json(meta_path, existing_data)

    macro_dep_path = f"{database_dir}/macro_deps.json"
    if macro_on is True:
        write_json(macro_dep_path, macro_symbols)

    print(f"\n{'='*60}")
    print(f"📁 Total files with metadata: {len(file_symbols)}")
    #print(f"📊 Total symbols: {sum(len(syms) for syms in file_symbols.values())}")
    print(f"{'='*60}\n")

    return macro_dep_path  # metadata, database_path, 


def get_is_function(item):
    kind = item['kind']
    if kind == "macro_function":
        return True
    else:
        return False


# A macro that is not a function-like macro (isFunctionLike() is false) and has 0 tokens (getNumTokens() == 0). Specifically, this refers to a #define with no value.
def get_is_flag(item):
    kind = item.get('kind', 'macro')
    if kind == "macro_flag":
        return True
    else:
        return False


def get_is_const(item):
    is_const = item.get('is_const', False)
    return is_const



def process_generate_macro_usage_metadata(entry, analyzer_path, compile_dir):
    """Function to process a single file (for parallel execution)"""
    
    source_file = entry.get('file')
    if not source_file:
        return {
            'source_file': None,
            'symbols': None,
            'error': None,
            'skip': True
        }
    
    # Skip assembly files
    if source_file.endswith(('.S', '.s', '.asm', '.ASM')):
        return {
            'source_file': source_file,
            'symbols': None,
            'error': None,
            'skip': True
        }
    
    directory = entry.get('directory', str(compile_dir))
    
    # Get compile arguments
    arguments = entry.get('arguments', [])
    if not arguments:
        command = entry.get('command', '')
        arguments = command.split()
    
    source_file_abs = source_file
    compile_args = []
    skip_next = False
    
    if len(arguments) > 1:
        for arg in arguments[1:]:
            if skip_next:
                skip_next = False
                continue
            
            if arg == '-o':
                skip_next = True
                continue
            
            if not os.path.isabs(arg):
                arg_abs = os.path.normpath(os.path.join(directory, arg))
            else:
                arg_abs = os.path.normpath(arg)
            
            if arg_abs != source_file_abs and not arg.endswith('.o'):
                compile_args.append(arg)
    
    cmd = [analyzer_path, source_file, "--"] + compile_args
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=directory
    )
    
    if result.returncode == 0:
        if result.stdout and result.stdout.strip():
            try:
                file_metadata = json.loads(result.stdout)
                file_symbols = file_metadata.get('macros', [])
                return {
                    'source_file': source_file,
                    'symbols': file_symbols,
                    'error': None,
                    'skip': False
                }
            except json.JSONDecodeError as e:
                return {
                    'source_file': source_file,
                    'symbols': None,
                    'error': {
                        'type': 'json_parse_error',
                        'message': str(e)
                    },
                    'skip': False
                }
        else:
            return {
                'source_file': source_file,
                'symbols': [],
                'error': None,
                'skip': False
            }
    else:
        return {
            'source_file': source_file,
            'symbols': None,
            'error': {
                'type': 'process_error',
                'returncode': result.returncode,
                'stderr': result.stderr,
                'stdout': result.stdout,
                'cmd': ' '.join(cmd),
                'cwd': directory
            },
            'skip': False
        }


def generate_macro_usage_metadata(target_dir, meta_dir, database_dir, independent_path, flag_path, compile_dir, compile_json, round_id):

    analyzer_path = "/home/ubuntu/c_parser/test2/build/analyzer"
    analyzer_path = "/home/ubuntu/c_parser/usage_macro_analyzer/build/analyzer"
    analyzer_path = "/home/ubuntu/c_parser/usage_macro_ref_analyzer/build/analyzer"

    # Path normalization
    """
    compile_dir = find_compile_commands_json(target_dir)
    compile_dir = Path(compile_dir)
    compile_json = compile_dir / "compile_commands.json"
    print(compile_json)
    convert_to_absolute_paths(compile_json)  # This is necessary!
    """

    # Check if compile_commands.json exists
    if not compile_json.exists():
        raise FileNotFoundError(
            f"compile_commands.json not found at: {compile_json}"
        )

    # Check if the analyzer tool exists
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(
            f"Analyzer tool not found at: {analyzer_path}\n"
            "Please build the analyzer first."
        )

    # Load compile_commands.json (for summary)
    with open(compile_json, 'r') as f:
        compile_commands = json.load(f)

    print(f"Processing {len(compile_commands)} files (batch mode)...")

    # === Plan B: Batch execution ===
    cmd = [analyzer_path, "-p", str(compile_dir)]
    output_path = Path(database_dir) / "analyzer_raw_output.json"

    with open(output_path, 'w') as outf:
        result = subprocess.run(
            cmd,
            stdout=outf,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=str(compile_dir)
        )

    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True, #check=False,
        cwd=str(compile_dir)
    )
    """

    if result.returncode != 0:
        print(f"❌ Analyzer failed (exit code: {result.returncode})")
        print(f"STDERR: {result.stderr}")
        print(f"Command: {' '.join(cmd)}")
        raise ValueError(f"❌ Analyzer error: {result.stderr}")

    # Parse JSON output
    all_symbols = []
    failed_files = []

    #if result.stdout.strip():
    if output_path.stat().st_size > 0:
        try:
            with open(output_path, 'r') as f:
                file_metadata = json.load(f)
            all_symbols = file_metadata.get('macros', [])
            for macro in all_symbols:
                definition = macro.get('definition', '')
                
                if ':' in definition:
                    parts = definition.split(':')
                    if len(parts) >= 3:
                        def_file_path = parts[0]
                        if not os.path.isabs(def_file_path):
                            compile_dir = os.path.abspath(compile_dir)
                            def_file_path = os.path.join(str(compile_dir), def_file_path)

                        def_file_path = os.path.normpath(def_file_path)
                        try:
                            start_line = int(parts[1])
                            col_num = int(parts[2])
                        except (ValueError, IndexError):
                            start_line = 0
                            col_num = 0
                        
                        definition = f"{def_file_path}:{start_line}:{col_num}"
                        macro['definition'] = definition

            del file_metadata  # Release immediately

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Error at line {e.lineno}, column {e.colno}")

            with open(output_path, 'r') as f:
                lines = f.read().split('\n')
            error_line = e.lineno - 1
            start = max(0, error_line - 3)
            end = min(len(lines), error_line + 4)
            print("--- Context ---")
            for i in range(start, end):
                marker = ">>>" if i == error_line else "   "
                line_content = lines[i][:150] if len(lines[i]) > 150 else lines[i]
                print(f"  {marker} L{i+1}: {line_content}")
            print("---------------")

            print(f"Debug saved: {output_path}")

            raise ValueError(f"❌ JSON parse error in batch output")

        """
        try:
            file_metadata = json.loads(result.stdout)
            all_symbols = file_metadata.get('macros', [])
        
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Error at line {e.lineno}, column {e.colno}")

            lines = result.stdout.split('\n')
            error_line = e.lineno - 1
            start = max(0, error_line - 3)
            end = min(len(lines), error_line + 4)
            print("--- Context ---")
            for i in range(start, end):
                marker = ">>>" if i == error_line else "   "
                line_content = lines[i][:150] if len(lines[i]) > 150 else lines[i]
                print(f"  {marker} L{i+1}: {line_content}")
            print("---------------")

            debug_path = "/tmp/debug_macro_batch_output.json"
            with open(debug_path, 'w') as f:
                f.write(result.stdout)
            print(f"Debug saved: {debug_path}")

            raise ValueError(f"❌ JSON parse error in batch output")
        """

    # Summary of processing results
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed in generate_macro_usage_metadata (batch mode)(batch mode) @round {round_id}")
    print(f"📊 Total symbols collected: {len(all_symbols)}")
    print(f"{'='*60}\n")

    
    # Wrote independent data independently here
    for item in all_symbols:
        definition = item.get('definition', '')
        start_line = item.get('start_line', None)
        end_line = item.get('end_line', None)

        item['is_independent'] = get_is_independent(item)


    # Create the overall metadata
    metadata = {'macros': all_symbols}

    # Create the database directory
    os.makedirs(database_dir, exist_ok=True)
    database_path = Path(database_dir) / "meta_macro_usage.json"
    write_json(str(database_path), metadata)
    print(f"Saved global metadata to: {database_path}")

    return all_symbols, database_path, metadata #, metadata, database_path


def update_macro_usage_metadata(all_symbols, meta_dir, independent_path, flag_path, compile_dir):

    # Classify symbols by file
    file_symbols = defaultdict(list)
    flag_macros = []
    independent_macros = []

    for item in all_symbols:
        definition = item.get('definition', '')
        start_line = item.get('start_line', None)
        end_line = item.get('end_line', None)

        # Extract file path from definition
        if ':' in definition:
            parts = definition.split(':')
            if len(parts) >= 3:
                file_path = parts[0]

                if not os.path.isabs(file_path):
                    # <built-in>
                    # <built-in>:370:9
                    # print(definition)
                    # print(file_path)
                    compile_dir = os.path.abspath(compile_dir)
                    file_path = os.path.join(str(compile_dir), file_path)

                file_path = os.path.normpath(file_path)

                try:
                    line_num = int(parts[1])
                    col_num = int(parts[2])
                except (ValueError, IndexError):
                    line_num = 0
                    col_num = 0

                is_function = get_is_function(item)
                is_flag = get_is_flag(item)
                is_independent = get_is_independent(item)
                is_const = get_is_const(item) # Overwritten to false if constant evaluation fails in even one place

                definition = f"{file_path}:{start_line}:{col_num}"
                meta_item = {
                    'kind': item.get('kind', 'macro'),
                    'name': item.get('name', ''),
                    'definition': definition,
                    'signature': item.get('signature', None),
                    'is_function': is_function,
                    'is_const': is_const, #False,
                    'is_independent': is_independent,
                    'is_flag': None, #is_flag,
                    'is_guard': False,
                    'is_guarded': False,  
                    'expanded_value': item.get('expanded_value', None),
                    'start_line': start_line,
                    'start_column': col_num,
                    'end_line': end_line,
                    'block_start': int(start_line),
                    'block_end': int(end_line),
                    'rust_code': {
                        'file_path': None,
                        'start_line': None,
                    },
                    'appearances': item.get('appearances', [])
                }
                if item.get('kind') == 'function':
                    meta_item['signature'] = item.get('signature', '')

                file_symbols[file_path].append(meta_item)

                if is_flag is True:
                    flag_macros.append({
                        'name': item.get('name', ''),
                        'file_path': file_path,
                        'start_line': start_line,
                        'end_line': end_line,
                    })

                if is_independent is True:
                    independent_macros.append({
                        'name': item.get('name', ''),
                        'file_path': file_path,
                        'start_line': start_line,
                        'end_line': end_line,
                    })

    # write_json(flag_path, flag_macros)  # modified
    write_json(independent_path, independent_macros)

    # Save metadata per file
    print(f"\nSaving per-file metadata...")
    for file_path, symbols in file_symbols.items():
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping non-existent file: {file_path}")
            continue

        meta_path = obtain_metadata(file_path, meta_dir, False, True, "def")
        meta_path = Path(meta_path)

        existing_data = {}
        if os.path.exists(meta_path):
            existing_data = read_json(meta_path)

        for symbol in symbols:
            name = symbol['name']
            s_line = symbol['start_line']
            item_key = f"{name}:{file_path}:{s_line}"

            if item_key not in existing_data:
                existing_data[item_key] = symbol

        write_json(meta_path, existing_data)

    print(f"\n{'='*60}")
    print(f"📁 Total files with usage metadata: {len(file_symbols)}")
    #print(f"📊 Total symbols: {sum(len(syms) for syms in file_symbols.values())}")
    print(f"{'='*60}\n")

    #return metadata, database_path


def get_is_independent(macro):
    """
    Determine whether a macro is a constant (independent) or a variable (dependent)
    
    Args:
        macro: A metadata dictionary of the macro
        
    Returns:
        bool: True = constant (independent), False = variable (dependent)
    """
    # Function-like macros are always treated as variables
    if macro.get('kind') == 'macro_function':
        return False
    
    # uses is empty → constant (does not reference other identifiers)
    uses = macro.get('uses', [])
    if not uses:
        return True
    
    # uses contains something → variable (depends on other identifiers)
    return False



# I think this information is correct, but the question is where to detect it.
def detect_independent_macros(unique_macros, independent_path):

    # independent/dependent determination
    independent_macros = []
    for macro in unique_macros.values():
        macro['is_independent'] = get_is_independent(macro)
        if macro['is_independent'] is True:
            independent_macros.append(macro)
    
    write_json(independent_path, independent_macros)

    return unique_macros



def generate_macro_metadata(target_dir, meta_dir, database_dir, independent_path, compile_dir, compile_json, round_id):
    """
    Specify the directory containing compile_commands.json, run the tool created in macro_analyzer.cpp in batch, extract macro metadata (JSON), and save it split by file.
    """
    macro_analyzer_path = "/home/ubuntu/macrust/macro_analyzer/build/macro_analyzer"

    """
    compile_dir = find_compile_commands_json(target_dir)
    compile_dir = Path(compile_dir)
    compile_json = compile_dir / "compile_commands.json"
    """

    if not compile_json.exists():
        raise FileNotFoundError(f"compile_commands.json not found at: {compile_json}")

    if not os.path.exists(macro_analyzer_path):
        raise FileNotFoundError(
            f"Macro analyzer tool not found at: {macro_analyzer_path}\n"
            "Please build the macro_analyzer first."
        )

    compile_commands = read_json(compile_json)

    if not compile_commands:
        raise ValueError("compile_commands.json is empty")

    # Remove duplicated entries
    print(f"Original compile_commands entries: {len(compile_commands)}")

    unique_commands = {}
    for entry in compile_commands:
        source_file = entry.get('file')
        if not source_file:
            continue

        if source_file.endswith(('.S', '.s', '.asm', '.ASM')):
            continue

        directory = entry.get('directory', str(compile_dir))

        if not os.path.isabs(source_file):
            abs_path = os.path.join(directory, source_file)
        else:
            abs_path = source_file

        try:
            normalized_path = os.path.realpath(abs_path)
        except:
            normalized_path = os.path.normpath(os.path.abspath(abs_path))

        if normalized_path not in unique_commands:
            unique_commands[normalized_path] = entry
        else:
            print(f"  ⚠️  Skipping duplicate: {source_file}")

    compile_commands = list(unique_commands.values())
    print(f"After deduplication: {len(compile_commands)} unique files")
    print(f"\nProcessing macros from {len(compile_commands)} files (batch mode)...")

    # Create a metadata directory.
    os.makedirs(meta_dir, exist_ok=True)

    # Plan B: Run All at Once
    abs_compile_dir = str(Path(compile_dir).resolve())
    cmd = [macro_analyzer_path, "-p", abs_compile_dir]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(compile_dir)
    )

    if result.returncode != 0:
        print(f"❌ Macro analyzer failed (exit code: {result.returncode})")
        print(f"STDERR: {result.stderr}")
        print(f"Command: {' '.join(cmd)}")
        raise ValueError(f"❌ Macro analyzer error: {result.stderr}")

    all_macros = []
    failed_files = []
    file_macros = defaultdict(list)

    if result.stdout.strip():
        try:
            file_metadata = json.loads(result.stdout)

            for macro in file_metadata.get('macros', []):
                definition = macro.get('definition', '')
                # all_macros.append(macro)

                if ':' in definition:
                    parts = definition.split(':')
                    if len(parts) >= 3:
                        def_file_path = parts[0]

                        if not os.path.isabs(def_file_path):
                            compile_dir = os.path.abspath(compile_dir)
                            def_file_path = os.path.join(str(compile_dir), def_file_path)

                        def_file_path = os.path.normpath(def_file_path)
                        
                        try:
                            start_line = int(parts[1])
                            col_num = int(parts[2])
                        except (ValueError, IndexError):
                            start_line = 0
                            col_num = 0
                        
                        definition = f"{def_file_path}:{start_line}:{col_num}"
                        macro['definition'] = definition

                        macro_name = macro.get('name', '')
                        end_line = macro.get('end_line')
                        item_key = f"{macro_name}:{def_file_path}:{start_line}"

                        use_data = macro.get('uses', [])
                        for use_item in use_data:
                            if 'definition' in use_item:
                                uf, ul, uc = parse_def_loc(use_item['definition'])
                                use_item['file_path'] = uf
                                use_item['start_line'] = ul

                        meta_item = {
                            'kind': macro.get('kind'),
                            'name': macro_name,
                            'definition': definition,
                            'start_line': start_line,
                            'start_column': col_num,
                            'end_line': end_line,
                            'end_column': None,
                            'block_start': int(start_line),
                            'block_end': int(end_line),
                            'parameters': macro.get('parameters', []),
                            'uses': use_data,
                            'file_path': def_file_path
                        }

                        file_macros[def_file_path].append((item_key, meta_item))
                
                all_macros.append(macro)

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Error at line {e.lineno}, column {e.colno}")

            lines = result.stdout.split('\n')
            error_line = e.lineno - 1
            start = max(0, error_line - 3)
            end = min(len(lines), error_line + 4)
            print("--- Context ---")
            for i in range(start, end):
                marker = ">>>" if i == error_line else "   "
                line_content = lines[i][:150] if len(lines[i]) > 150 else lines[i]
                print(f"  {marker} L{i+1}: {line_content}")
            print("---------------")

            debug_path = "/tmp/debug_macro_def_batch.json"
            with open(debug_path, 'w') as f:
                f.write(result.stdout)
            print(f"Debug saved: {debug_path}")

            raise ValueError(f"❌ JSON parse error in macro batch output")

    print(f"\n{'='*60}")
    print(f"✅ Successfully processed in generate_macro_metadata (batch mode) @round {round_id}")
    print(f"📊 Total macros collected: {len(all_macros)}")
    print(f"{'='*60}\n")

    # Remove duplicates
    unique_macros = {}
    for macro in all_macros:
        macro_name = macro['name']
        definition = macro.get('definition', '')
        key = f"{macro_name}:{definition}"
        # if macro_name not in unique_macros:
        #     unique_macros[macro_name] = macro
        if key not in unique_macros:
            unique_macros[key] = macro

    database_path = Path(database_dir) / "meta_macro.json"
    metadata = {"macros": list(unique_macros.values())}
    write_json(str(database_path), metadata)
    print(f"Saved global macro metadata to: {database_path}")

    return file_macros, database_path  #unique_macros, database_path


def update_macro_metadata(file_macros, meta_dir):

    print(f"\nSaving per-file macro metadata...")
    for file_path, macros_list in file_macros.items():
        meta_path = obtain_metadata(file_path, meta_dir, False, True, "def")
        if meta_path is None:
            continue

        meta_path = Path(meta_path)

        if os.path.exists(meta_path):
            try:
                loaded_data = read_json(meta_path)
                if isinstance(loaded_data, list):
                    existing_data = {}
                    for item in loaded_data:
                        name = item.get('name', '')
                        sl = item.get('start_line', 0)
                        item_file = item.get('file_path', file_path)
                        key = f"{name}:{item_file}:{sl}"
                        existing_data[key] = item
                elif isinstance(loaded_data, dict):
                    existing_data = loaded_data
                else:
                    existing_data = {}
            except:
                existing_data = {}
        else:
            existing_data = {}

        for item_key, macro_item in macros_list:
            if item_key not in existing_data:
                existing_data[item_key] = macro_item

        write_json(meta_path, existing_data)

    print(f"\n{'='*60}")
    print(f"📁 Total files with macros: {len(file_macros)}")
    #print(f"📊 Total macros: {len(all_macros)}")
    #print(f"📊 Unique macros: {len(unique_macros)}")
    print(f"{'='*60}\n")


def merge_appearances_with_uses(target_dir, meta_dir, database_dir, macros_usage_data):
    """Add appearances that fall within the start_line~end_line range of metadata to uses"""
    
    # Group appearances by file path
    appearances_by_file = defaultdict(list)
    
    for macro in macros_usage_data.get('macros', []):
        name = macro.get('name')
        if macro.get('file_path') == "": # Cases where things like printf are included
            continue
        for appearance in macro.get('appearances', []):
            # appearance format: "/path/to/file.c:line:column"
            parts = appearance.rsplit(':', 2)
            if len(parts) >= 2:
                file_path = parts[0]
                line = int(parts[1])
                column = int(parts[2])

                appearances_by_file[file_path].append({
                    "definition": macro.get('definition'),
                    "file_path": macro.get('file_path'),
                    "start_line": macro.get('start_line'),
                    "end_line": macro.get('end_line'),
                    'name': name,
                    'line': line,
                    'column': column,
                    #'appearances': appearances
                })

    
    meta_files = get_all_files(meta_dir)
    #print(meta_files)

    for meta_path in meta_files:
        #print(meta_path)
        metadata = read_json(meta_path)
        modified = False
        
        for key, meta in metadata.items():
            # start_line = meta.get('start_line')
            # end_line = meta.get('end_line')
            start_line = meta.get('block_start')
            end_line = meta.get('block_end')
            existing_uses = meta.get('uses', [])
            
            
            # Extract file path from the metadata's definition
            if 'definition' in meta:
                definition = meta.get('definition', '')
                def_parts = definition.rsplit(':', 2)
                # if len(def_parts) < 2:
                #     continue
                file_path = def_parts[0]
            else:
                file_path = meta.get('file_path')
            
            # Check only appearances for the relevant file
            file_appearances = appearances_by_file.get(file_path, [])

            new_uses = []
            for item in file_appearances:
                if start_line <= item['line'] <= end_line:
                    #new_uses.append(f"{item['name']}:{item['appearances']}")
                    line = item['line']
                    column = item['column']
                    usage_location = f"{file_path}:{line}:{column}"
                    new_uses.append({
                        "name" : item['name'],
                        "definition" : f"{item.get('definition')}",
                        "file_path": item.get('file_path'),
                        "start_line": item.get('start_line'),
                        "end_line": item.get('end_line'),
                        #"use_file_path" : file_path,
                        "usage_location" : usage_location,
                        #"line" : item['line']
                    })

            
            if new_uses:
                meta['uses'] = existing_uses + new_uses
                modified = True
        
        if modified:
            write_json(meta_path, metadata)



def resolve_addresses(binary_path: str, addresses: list[str]) -> dict[str, str]:
    """Batch-convert addresses to function names using addr2line"""
    if not addresses:
        return {}

    proc = subprocess.run(
        ["addr2line", "-f", "-e", binary_path] + addresses,
        capture_output=True, text=True
    )

    lines = proc.stdout.strip().split("\n")
    result = {}
    # addr2line -f returns pairs of two lines: function_name\nfile:line
    for i in range(0, len(lines), 2):
        func_name = lines[i] if i < len(lines) else "??"
        location = lines[i + 1] if i + 1 < len(lines) else "??"
        addr = addresses[i // 2]
        result[addr] = (func_name, location)
        #result[addr] = func_name

    return result



##################################
### Tracking part
##################################


def make_relative(location, base_dir):
    if location == "??" or location == "??:?":
        return location
    try:
        file_part, line_part = location.rsplit(":", 1)
        rel = os.path.relpath(file_part, base_dir)
        return f"{rel}:{line_part}"
    except:
        return location


def parse_trace(trace_path: str, base_dir: str, output_path: str, is_rust: bool):  # = "call_tree.txt"# binary_path: str, 
    """Parse the trace log and output the call tree to a txt file"""

    MAX_OUTPUT_LINES = 5_000_000

    entries = []
    unique_addrs = OrderedDict()
    maps = []
    base_addrs = {}

    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# map "):
                map_line = line[6:]
                parts = map_line.split()
                if len(parts) >= 6:
                    addr_range = parts[0].split("-")
                    start = int(addr_range[0], 16)
                    end = int(addr_range[1], 16)
                    file_offset = int(parts[2], 16)
                    path = parts[-1]
                    maps.append((start, end, path, file_offset))
                    if path not in base_addrs and file_offset == 0:
                        base_addrs[path] = start
                continue

            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            typ, tid_str, depth_str, func_addr, caller_addr = parts
            if typ not in ("E", "X"):
                continue

            tid = int(tid_str)
            entries.append((typ, tid, int(depth_str), func_addr, caller_addr))
            unique_addrs[func_addr] = None
            unique_addrs[caller_addr] = None

    # Group addresses by binary
    addr_list = list(unique_addrs.keys())
    binary_groups = {}

    for addr_hex in addr_list:
        addr_int = int(addr_hex, 16)
        matched = False
        for start, end, path, file_offset in maps:
            if start <= addr_int < end:
                base = base_addrs.get(path, start)
                offset = addr_int - base
                if path not in binary_groups:
                    binary_groups[path] = {}
                binary_groups[path][addr_hex] = hex(offset)
                matched = True
                break
        if not matched:
            if "__unmatched__" not in binary_groups:
                binary_groups["__unmatched__"] = {}
            binary_groups["__unmatched__"][addr_hex] = addr_hex

    # Resolve using addr2line for each binary
    resolved = {}
    for bin_path, addr_map in binary_groups.items():
        if bin_path == "__unmatched__":
            for orig in addr_map:
                resolved[orig] = ("??", "??")
            continue

        offsets = list(addr_map.values())
        originals = list(addr_map.keys())
        raw = resolve_addresses(bin_path, offsets)

        for orig, offset in zip(originals, offsets):
            resolved[orig] = raw.get(offset, ("??", "??"))

    # Demangle Rust symbols
    mangled_names = [n for n, _ in resolved.values() if n.startswith("_R") or n.startswith("_ZN")]
    if mangled_names:
        proc = subprocess.run(
            ["rustfilt"],
            input="\n".join(mangled_names),
            capture_output=True, text=True
        )
        demangled = proc.stdout.strip().split("\n")
        demangle_map = dict(zip(mangled_names, demangled))

        for addr in resolved:
            name, loc = resolved[addr]
            if name in demangle_map:
                resolved[addr] = (demangle_map[name], loc)

    # Separate entries by thread
    thread_entries = {}
    for typ, tid, depth, func_addr, caller_addr in entries:
        if tid not in thread_entries:
            thread_entries[tid] = []
        thread_entries[tid].append((typ, depth, func_addr, caller_addr))

    # Output call tree per thread (with exit inference)
    total_entries = 0
    line_count = 0 
    truncated = False

    #with open(output_path, "w") as out:
    with open(output_path, "w", buffering=8*1024*1024) as out:
        for tid in sorted(thread_entries.keys()):
            thread_data = thread_entries[tid]
            total_entries += len(thread_data)

            if len(thread_entries) > 1:
                out.write(f"\n=== Thread {tid} ===\n\n")

            stack = []

            if is_rust is False:
                for typ, depth, func_addr, caller_addr in thread_data:
                    if line_count >= MAX_OUTPUT_LINES: 
                        truncated = True 
                        break

                    func_name, location = resolved.get(func_addr, ("??", "??"))
                    if base_dir is not None:
                        location = make_relative(location, base_dir)

                    if typ == "E":
                        while stack and stack[-1][0] >= depth:
                            d, fn, loc = stack.pop()
                            indent = "  " * d
                            out.write(f"{indent}<- {fn}  [{loc}]\n")

                        indent = "  " * depth
                        out.write(f"{indent}-> {func_name}  [{location}]\n")
                        stack.append((depth, func_name, location))

                    elif typ == "X":
                        while stack and stack[-1][0] > depth:
                            d, fn, loc = stack.pop()
                            indent = "  " * d
                            out.write(f"{indent}<- {fn}  [{loc}]\n")

                        if stack and stack[-1][0] == depth:
                            d, fn, loc = stack.pop()
                            indent = "  " * d
                            out.write(f"{indent}<- {fn}  [{loc}]\n")
                        else:
                            indent = "  " * depth
                            out.write(f"{indent}<- {func_name}  [{location}]\n")
            
            else:
                for typ, depth, func_addr, caller_addr in thread_data:
                    if line_count >= MAX_OUTPUT_LINES:
                        truncated = True
                        break 

                    if typ != "E":
                        continue
                    func_name, location = resolved.get(func_addr, ("??", "??"))
                    if base_dir is not None:
                        location = make_relative(location, base_dir)
                    out.write(f"-> {func_name}  [{location}]\n")
                    line_count += 1 

            if truncated: 
                break

            while stack:
                d, fn, loc = stack.pop()
                indent = "  " * d
                out.write(f"{indent}<- {fn}  [{loc}]\n")

    print(f"Entry count: {total_entries}")
    print(f"Thread count: {len(thread_entries)}")
    print(f"Unique function count: {len(set(n for n, _ in resolved.values()) - {'??'})}")
    print(f"Output: {output_path}")

    if truncated:         
        print(f"WARNING: Output truncated at {MAX_OUTPUT_LINES} lines")

"""
cargo install rustfilt
"""


def find_binaries(workspace):
    """Search for ELF binaries in the workspace"""
    binaries = []
    for root, dirs, files in os.walk(workspace):
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "rb") as fh:
                    if fh.read(4) == b"\x7fELF":
                        binaries.append(path)
            except (PermissionError, IsADirectoryError, OSError):
                continue
    return binaries


def run_with_trace(workspace, trace_output, tracer_so_path, args=None):

    binaries = find_binaries(workspace)
    if not binaries:
        print("[!] No ELF binaries found")
        return None

    print(f"[*] Detected binaries: {binaries}")
    binary_path = binaries[0]  # Use the first binary

    env = os.environ.copy()
    env["LD_PRELOAD"] = os.path.abspath(tracer_so_path)
    env["TRACE_OUTPUT"] = trace_output

    cmd = [binary_path]
    if args:
        cmd += args

    subprocess.run(cmd, env=env)

    return binary_path



if __name__ == "__main__":

    workspace_dir = "/home/ubuntu/c_parser/sample"
    trace_output = "/home/ubuntu/c_parser/sample/trace.log"
    tracer_so = "/home/ubuntu/c_parser/c_parser_api/libtracer.so"
    binary_path = "/home/ubuntu/c_parser/sample/test_ffi"

    """
    env = os.environ.copy()
    env["LD_PRELOAD"] = tracer_so
    env["TRACE_OUTPUT"] = trace_output
    subprocess.run([binary_path], env=env)
    """

    trace_output = "/home/ubuntu/macrust/trans_re_0000/bst/genifai_results/test1_trace.log"
    parse_trace(trace_output, "/home/ubuntu/macrust/trans_re_0000", "call_tree.txt") # binary_path: str, 


    workspace_dir = "/home/ubuntu/c_parser/sample"
    trace_output = "trace.log"

    binary_path = run_with_trace(
        workspace=workspace_dir,
        trace_output=trace_output,
        tracer_so_path = "/home/ubuntu/c_parser/c_parser_api/libtracer.so",
        # build_script="/home/ubuntu/allrust/workspace_0000_zopfli/run_all.sh",
        # test_script="/home/ubuntu/allrust/workspace_0000_zopfli/zopfli/run_test.sh",
        # output="trace.txt",
    )

    if binary_path:
        parse_trace(binary_path, trace_output, "call_tree.txt")


"""
# trace on
LD_PRELOAD=/path/to/libtracer.so ./program

# trace off (Execute the same binary as usual)
./program
"""

"""
ls -la /usr/local/lib/libtracer.so
sudo cp /home/ubuntu/c_parser/c_parser_api/libtracer.so /usr/local/lib/
sudo ldconfig
"""