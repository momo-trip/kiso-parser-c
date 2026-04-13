# kiso-parser-c

A suite of static analysis tools for C source code. Using Clang LibTooling, it extracts symbols (functions, variables, types, macros, etc.), their definitions, dependencies, and usage locations, and outputs them in JSON format.

## Requirements

* Ubuntu (tested on Ubuntu 22.04+)
* LLVM/Clang 19 (including custom-built versions)
* CMake 3.16+
* C++17-compatible compiler
* Python 3.10+ (for API server usage)

## Directory Structure

```
c_parser/
├── usage_analyzer/          # Usage analysis tool
├── usage_macro_ref_analyzer/# Macro reference analysis tool
├── include_finder/          # Include dependency analysis tool
├── c_parser_api/            # Python API
├── llvm-project/            # LLVM/Clang source (with custom patches applied)
├── llvm-custom/             # Prebuilt custom LLVM
├── clang-modifications.patch # Custom patch for Clang
├── download_clang.sh        # LLVM download script
├── pyproject.toml           # Python package configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

### 1. Build LLVM/Clang

```bash
# Download LLVM source and apply custom patch
./download_clang.sh
```

### 2. Build Analysis Tools

Build each tool using CMake inside its directory.

```bash
# Example: build usage_analyzer
cd usage_analyzer
./build.sh

# Example: build usage_macro_analyzer
cd usage_macro_analyzer
./build.sh 
```

### 3. Python API

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Usage

### Symbol Analysis - usage_analyzer

Extracts symbols such as functions, variables, structs, typedefs, enums, and macros from source code, and outputs their dependencies (`uses`) in JSON.

**Analyze a single file:**

```bash
./usage_analyzer/build/analyzer test.c
```

**Analyze a project using compile_commands.json:**

```bash
./usage_analyzer/build/analyzer -p /path/to/build_dir
```

The directory specified with `-p` must contain `compile_commands.json`. For CMake projects, it can be generated with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.


**Example output:**

```json
{
  "symbols": [
    {
      "kind": "function",
      "name": "main",
      "signature": "int main(int argc, char **argv)",
      "definition": "/path/to/test.c:10:5",
      "file_path": "/path/to/test.c",
      "start_line": 10,
      "end_line": 25,
      "uses": [
        {
          "kind": "function",
          "name": "printf",
          "definition": "/usr/include/stdio.h:332:12",
          "usage_location": "/path/to/test.c:15:5"
        }
      ]
    }
  ]
}
```

**Types of extracted symbols:**

| kind                         | Description                    |
| ---------------------------- | ------------------------------ |
| `function`                   | Function definition            |
| `function_decl`              | Function prototype declaration |
| `global_var`                 | Global variable definition     |
| `global_var_decl`            | extern declaration             |
| `struct` / `union`           | Struct/union definition        |
| `struct_decl` / `union_decl` | Forward declaration            |
| `typedef`                    | Type alias                     |
| `enum`                       | Enum definition                |
| `enum_constant`              | Enum constant                  |
| `field`                      | Struct field                   |
| `macro`                      | Macro definition               |

### Detailed Macro Analysis - usage_macro_analyzer

In addition to macro definitions, expansions, and usage locations, it resolves symbols (`uses`) referenced within macro bodies. It adopts a two-phase approach combining preprocessing-time analysis and AST traversal with deferred resolution.

**Analyze a single file:**

```bash
./usage_macro_analyzer/build/usage_macro_analyzer test.c
```

**Analyze a project using compile_commands.json:**

```bash
./usage_macro_analyzer/build/usage_macro_analyzer -p /path/to/build_dir
```


**Example output:**

```json
{
  "macros": [
    {
      "kind": "macro_function",
      "name": "MAX",
      "definition": "/path/to/test.c:3:9",
      "file_path": "/path/to/test.c",
      "start_line": 3,
      "end_line": 3,
      "is_const": false,
      "expanded_value": "( ( a ) > ( b ) ? ( a ) : ( b ) )",
      "parameters": ["a", "b"],
      "appearances": [
        "/path/to/test.c:20:15",
        "/path/to/test.c:25:10"
      ],
      "uses": [...]
    }
  ]
}
```

**Macro classification:**

| kind             | Description                             |
| ---------------- | --------------------------------------- |
| `macro`          | Object macro (`#define FOO 42`)         |
| `macro_function` | Function macro (`#define MAX(a,b) ...`) |
| `macro_flag`     | Flag macro (`#define DEBUG`)            |



### Include Dependency Analysis - include_finder

Analyzes `#include` dependencies of source files.

```bash
./include_finder/build/include_finder test.c
```

### Python API - c_parser_api

Exposes the above tools as an Python API.
